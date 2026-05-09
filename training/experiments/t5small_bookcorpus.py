from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from beta_paths import core_args, default_cache_dir, default_dataset_dir, resolve_repo_path, save_result
from torch_accel import bf16_autocast, build_adamw, use_cuda_bf16
from init_batch_probe import build_probe_payload, count_valid_targets, write_probe_payload


RESULT_PREFIX = "t5small_bookcorpus"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def random_segmentation(num_items: int, num_segments: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.arange(num_items - 1) < (num_segments - 1)
    rng.shuffle(mask)
    first_in_segment = np.pad(mask, [[1, 0]])
    segment_id = np.cumsum(first_in_segment)
    _, counts = np.unique(segment_id, return_counts=True)
    return counts


def random_spans_noise_mask(length: int, noise_density: float, mean_noise_span_length: float, rng: np.random.Generator) -> np.ndarray:
    num_noise_tokens = int(np.round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans, rng)
    nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans, rng)

    interleaved = np.reshape(np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2])
    span_starts = np.cumsum(interleaved)[:-1]
    span_start_indicator = np.zeros((length,), dtype=bool)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    return is_noise


def create_sentinel_ids(mask_indices: np.ndarray, tokenizer) -> np.ndarray:
    mask_indices = mask_indices.astype(np.int32, copy=False)
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]
    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    sentinel_ids = np.where(sentinel_ids != 0, tokenizer.vocab_size - sentinel_ids, 0)
    sentinel_ids -= mask_indices - start_indices
    return sentinel_ids


def filter_input_ids(input_ids: np.ndarray, sentinel_ids: np.ndarray, eos_token_id: int) -> np.ndarray:
    combined = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
    filtered = combined[combined >= 0]
    return np.concatenate([filtered, np.array([eos_token_id], dtype=np.int32)])


class BookCorpusT5Dataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        *,
        max_input_length: int,
        noise_density: float,
        mean_noise_span_length: float,
        seed: int,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_input_length = int(max_input_length)
        self.noise_density = float(noise_density)
        self.mean_noise_span_length = float(mean_noise_span_length)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        text = str(self.dataset[index]["text"]).strip()
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_input_length,
        )

        if len(token_ids) < 8:
            filler = self.tokenizer.encode(
                "This book passage is too short for denoising.",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_input_length,
            )
            token_ids = filler

        if len(token_ids) >= self.max_input_length:
            token_ids = token_ids[: self.max_input_length]

        token_ids_np = np.array(token_ids, dtype=np.int32)
        rng = np.random.default_rng(self.seed + index)
        noise_mask = random_spans_noise_mask(
            len(token_ids_np),
            noise_density=self.noise_density,
            mean_noise_span_length=self.mean_noise_span_length,
            rng=rng,
        )
        input_sentinel = create_sentinel_ids(noise_mask[None, :], self.tokenizer)
        label_sentinel = create_sentinel_ids((~noise_mask)[None, :], self.tokenizer)

        input_ids = filter_input_ids(token_ids_np, input_sentinel[0], self.tokenizer.eos_token_id)
        label_ids = filter_input_ids(token_ids_np, label_sentinel[0], self.tokenizer.eos_token_id)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def collate_t5_batch(features, tokenizer):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    max_input = max(item["input_ids"].shape[0] for item in features)
    max_label = max(item["labels"].shape[0] for item in features)

    input_ids = torch.full((len(features), max_input), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(features), max_input), dtype=torch.long)
    labels = torch.full((len(features), max_label), -100, dtype=torch.long)

    for i, item in enumerate(features):
        src = item["input_ids"]
        tgt = item["labels"]
        input_ids[i, : src.shape[0]] = src
        attention_mask[i, : src.shape[0]] = 1
        labels[i, : tgt.shape[0]] = tgt

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def weighted_mean(total_weighted: float, total_count: int) -> float:
    return total_weighted / max(total_count, 1)


def evaluate_model(model, val_loader, device, use_bf16: bool, *, max_val_batches: int) -> float:
    model.eval()
    val_weighted_loss = 0.0
    val_target_tokens = 0
    with torch.no_grad():
        val_batches_seen = 0
        for batch in val_loader:
            if val_batches_seen >= max_val_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            valid_tokens = int((labels != -100).sum().item())
            with bf16_autocast(device, use_bf16):
                outputs = model(**batch)
            if valid_tokens > 0:
                val_weighted_loss += float(outputs.loss.item()) * valid_tokens
                val_target_tokens += valid_tokens
            val_batches_seen += 1
    return weighted_mean(val_weighted_loss, val_target_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="T5-small denoising pretraining on BookCorpus with beta sweep support.")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--results_subdir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--total_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--noise_density", type=float, default=0.15)
    parser.add_argument("--mean_noise_span_length", type=float, default=3.0)
    parser.add_argument("--validation_docs", type=int, default=20000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--probe_init_batch_loss", action="store_true")
    parser.add_argument("--probe_output_subdir", type=str, default="init_batch_loss_probe")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    args = parser.parse_args()

    dataset_cache_dir = resolve_repo_path(args.dataset_cache_dir, default_dataset_dir("bookcorpus"))
    hf_cache_dir = resolve_repo_path(args.hf_cache_dir, default_cache_dir("huggingface") / RESULT_PREFIX)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = use_cuda_bf16(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    dataset_dict = load_from_disk(str(dataset_cache_dir))
    if "train" not in dataset_dict:
        raise ValueError(f"BookCorpus dataset at {dataset_cache_dir} must contain a train split")
    full_train = dataset_dict["train"]
    total_docs = len(full_train)
    validation_docs = min(max(1000, int(args.validation_docs)), max(1000, total_docs // 100))
    if total_docs <= validation_docs + 1:
        raise ValueError("BookCorpus dataset too small to create train/validation split")

    val_raw = full_train.select(range(validation_docs))
    train_raw = full_train.select(range(validation_docs, total_docs))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=str(hf_cache_dir),
        clean_up_tokenization_spaces=False,
    )
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Run start | dataset=bookcorpus beta={args.beta} seed={args.seed}")
    log(
        f"Config | batch_size={args.batch_size} total_batch_size={args.total_batch_size} "
        f"eval_batch_size={args.eval_batch_size} steps={args.steps} wd={args.wd}"
    )
    log(f"Paths | dataset_cache_dir={dataset_cache_dir} hf_cache_dir={hf_cache_dir}")
    log(f"Device | device={device} | bf16={use_bf16}")

    train_dataset = BookCorpusT5Dataset(
        train_raw,
        tokenizer,
        max_input_length=args.max_input_length,
        noise_density=args.noise_density,
        mean_noise_span_length=args.mean_noise_span_length,
        seed=args.seed,
    )
    val_dataset = BookCorpusT5Dataset(
        val_raw,
        tokenizer,
        max_input_length=args.max_input_length,
        noise_density=args.noise_density,
        mean_noise_span_length=args.mean_noise_span_length,
        seed=args.seed + 10_000_000,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=str(hf_cache_dir))
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)

    collate_fn = lambda features: collate_t5_batch(features, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if args.probe_init_batch_loss:
        t_probe = time.time()
        batch = next(iter(train_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        model.train()
        with torch.no_grad():
            with bf16_autocast(device, use_bf16):
                outputs = model(**batch)
        payload = build_probe_payload(
            args=args,
            case_id="t5small_bookcorpus",
            label="T5-small + BookCorpus",
            split="val",
            family="language",
            loss_value=float(outputs.loss.item()),
            batch=batch,
            valid_targets=count_valid_targets(batch["labels"]),
            model=model,
            meta={
                "dataset_cache_dir": str(dataset_cache_dir),
                "hf_cache_dir": str(hf_cache_dir),
                "train_docs": int(len(train_raw)),
                "validation_docs_reserved": int(validation_docs),
                "model_name": args.model_name,
            },
            use_bf16=use_bf16,
            device=device,
            start_time=t_probe,
        )
        out_path = write_probe_payload(args, RESULT_PREFIX, payload)
        log(f"Saved probe to {out_path}")
        return

    if args.total_batch_size % args.batch_size != 0:
        raise ValueError(
            f"total_batch_size ({args.total_batch_size}) must be divisible by batch_size ({args.batch_size})"
        )
    grad_accum = args.total_batch_size // args.batch_size

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "layer_norm" in name.lower() or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer, fused_adamw = build_adamw(
        [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
        device=device,
        lr=args.lr,
        betas=(args.beta, args.beta),
    )

    total_steps = max(1, int(args.steps))
    planned_micro_batches = total_steps * grad_accum
    max_train_batches = (
        planned_micro_batches
        if int(args.max_train_batches) <= 0
        else min(planned_micro_batches, int(args.max_train_batches))
    )
    max_val_batches = len(val_loader) if int(args.max_val_batches) <= 0 else min(len(val_loader), int(args.max_val_batches))
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    eval_every = max(1, int(args.eval_every))

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
    }

    train_iter = iter(train_loader)
    global_step = 0
    t_start = time.time()
    log(
        f"Data | train_docs={len(train_raw)} val_docs={len(val_raw)} "
        f"val_batches={len(val_loader)} effective_train_batches={max_train_batches} "
        f"effective_val_batches={max_val_batches} grad_accum={grad_accum}"
    )
    log(
        f"Optimizer | adamw_fused={fused_adamw} total_steps={total_steps} "
        f"warmup_steps={warmup_steps} eval_every={eval_every}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    interval_weighted_loss = 0.0
    interval_target_tokens = 0

    for batch_idx in range(1, max_train_batches + 1):
        if global_step >= total_steps:
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        valid_tokens = int((labels != -100).sum().item())

        with bf16_autocast(device, use_bf16):
            outputs = model(**batch)
            loss = outputs.loss

        if valid_tokens > 0:
            interval_weighted_loss += float(loss.item()) * valid_tokens
            interval_target_tokens += valid_tokens

        (loss / grad_accum).backward()

        if batch_idx % grad_accum == 0 or batch_idx == max_train_batches:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if global_step < warmup_steps:
                lr = args.lr * float(global_step + 1) / float(max(1, warmup_steps))
            else:
                t = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                lr = float(args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * t)))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            should_eval = (
                global_step % eval_every == 0
                or global_step >= total_steps
                or batch_idx >= max_train_batches
            )
            if should_eval:
                train_loss = weighted_mean(interval_weighted_loss, interval_target_tokens)
                val_loss = evaluate_model(model, val_loader, device, use_bf16, max_val_batches=max_val_batches)
                history["step"].append(global_step)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                log(
                    f"Eval | step={global_step}/{total_steps} | train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | time={time.time() - t_start:.1f}s"
                )
                interval_weighted_loss = 0.0
                interval_target_tokens = 0
                model.train()

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    log(
        f"Finished | train_loss={final_train_loss:.4f} | val_loss={final_val_loss:.4f} | "
        f"optimizer_steps={global_step}/{total_steps} | time={time.time() - t_start:.1f}s"
    )

    payload = {
        "args": {
            **core_args(args),
            "total_batch_size": int(args.total_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "min_lr": float(args.min_lr),
            "wd": float(args.wd),
            "max_input_length": int(args.max_input_length),
            "noise_density": float(args.noise_density),
            "mean_noise_span_length": float(args.mean_noise_span_length),
            "validation_docs": int(validation_docs),
            "eval_every": int(eval_every),
        },
        "history": history,
        "time_sec": float(time.time() - t_start),
        "meta": {
            "dataset": "bookcorpus",
            "task": "seq2seq_denoising_pretraining",
            "optimizer": "adamw",
            "model": args.model_name,
            "dataset_cache_dir": str(dataset_cache_dir),
            "hf_cache_dir": str(hf_cache_dir),
            "train_docs": int(len(train_raw)),
            "val_docs": int(len(val_raw)),
        },
    }
    out_path = save_result(RESULT_PREFIX, args, payload)
    log(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
