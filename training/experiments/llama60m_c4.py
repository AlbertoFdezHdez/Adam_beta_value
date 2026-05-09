from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import time
from pathlib import Path

import datasets
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from beta_paths import default_cache_dir, resolve_repo_path, save_result
from torch_accel import bf16_autocast, build_adam, use_cuda_bf16


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENTS_ROOT = SCRIPT_PATH.parent
CODE_ROOT = EXPERIMENTS_ROOT.parent
DEFAULT_MODEL_CONFIG = CODE_ROOT / "vendor" / "alphadecay_llama" / "configs" / "llama_60m.json"

DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "en"
RESULT_PREFIX = "llama60m_c4"


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size: int, max_length: int):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_data = iter(self.data)
        else:
            iter_data = itertools.islice(self.data, worker_info.id, None, worker_info.num_workers)

        batch = []
        for example in iter_data:
            tokenized = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized)
            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    @staticmethod
    def _format_batch(batch):
        return {
            "input_ids": torch.stack([item["input_ids"].squeeze(0) for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"].squeeze(0) for item in batch]),
        }


def batch_iterator(dataset, batch_size: int):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            yield {
                "input_ids": torch.stack([torch.tensor(item["input_ids"]).long() for item in batch]),
                "attention_mask": torch.stack([torch.tensor(item["attention_mask"]).long() for item in batch]),
            }
            batch = []
    if batch:
        yield {
            "input_ids": torch.stack([torch.tensor(item["input_ids"]).long() for item in batch]),
            "attention_mask": torch.stack([torch.tensor(item["attention_mask"]).long() for item in batch]),
        }


def cosine_with_warmup_lambda(current_step: int, *, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if total_steps <= 0:
        return 1.0
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def build_cosine_scheduler(optimizer, *, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup_lambda(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_input_ids_for_vocab(input_ids: torch.Tensor, vocab_size: int, unk_token_id: int | None) -> torch.Tensor:
    if vocab_size <= 0:
        return input_ids
    safe_ids = input_ids.clone()
    invalid_mask = (safe_ids < 0) | (safe_ids >= vocab_size)
    if invalid_mask.any():
        replacement = 0 if unk_token_id is None or int(unk_token_id) < 0 else int(unk_token_id)
        safe_ids[invalid_mask] = replacement
    return safe_ids


def prepare_val_iterator(tokenizer, hf_cache_dir: Path, eval_batch_size: int, max_length: int):
    def preprocess_batched(batch):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    val_data = datasets.load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="validation",
        streaming=True,
        cache_dir=str(hf_cache_dir),
    )
    val_data = val_data.shuffle(seed=42)
    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    return batch_iterator(val_data_mapped, eval_batch_size)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    tokenizer,
    eval_batch_size: int,
    max_length: int,
    target_eval_tokens: int,
    device: torch.device,
    use_bf16: bool,
    hf_cache_dir: Path,
) -> tuple[float, float, int]:
    model.eval()
    tokenizer_pad_id = tokenizer.pad_token_id
    tokenizer_unk_id = tokenizer.unk_token_id
    model_vocab_size = int(model.config.vocab_size)

    total_loss_weighted = 0.0
    total_tokens = 0
    iterator = prepare_val_iterator(
        tokenizer=tokenizer,
        hf_cache_dir=hf_cache_dir,
        eval_batch_size=eval_batch_size,
        max_length=max_length,
    )

    for batch in iterator:
        if total_tokens >= target_eval_tokens:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["input_ids"] = sanitize_input_ids_for_vocab(
            batch["input_ids"],
            vocab_size=model_vocab_size,
            unk_token_id=tokenizer_unk_id,
        )
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer_pad_id] = -100
        with bf16_autocast(device, use_bf16):
            outputs = model(**batch, labels=labels)
        valid_tokens = int((labels != -100).sum().item())
        if valid_tokens <= 0:
            continue
        total_loss_weighted += float(outputs.loss.item()) * valid_tokens
        total_tokens += valid_tokens

    mean_loss = total_loss_weighted / max(total_tokens, 1)
    ppl = float(math.exp(min(mean_loss, 50.0)))
    model.train()
    return mean_loss, ppl, total_tokens


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama60M on C4 with beta sweep support.")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=1_000)
    parser.add_argument("--status_every", type=int, default=50)
    parser.add_argument("--target_eval_tokens", type=int, default=10_000_000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--model_config_path", type=str, default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--results_subdir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    model_config_path = resolve_repo_path(args.model_config_path, DEFAULT_MODEL_CONFIG)
    hf_cache_root = resolve_repo_path(args.hf_cache_dir, default_cache_dir("huggingface") / "llama60m_c4")
    run_cache_dir = hf_cache_root / f"{RESULT_PREFIX}_bs{args.batch_size}_beta{args.beta:.5f}_seed{args.seed}"
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(run_cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(run_cache_dir / "datasets")

    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = use_cuda_bf16(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    log(f"Run start | dataset=C4(en) beta={args.beta} seed={args.seed}")
    log(
        f"Config | batch_size={args.batch_size} total_batch_size={args.total_batch_size} "
        f"eval_batch_size={args.eval_batch_size} steps={args.steps} wd={args.wd}"
    )
    log(f"Paths | hf_cache_root={hf_cache_root} run_cache_dir={run_cache_dir} model_config_path={model_config_path}")
    log(f"Device | device={device} | bf16={use_bf16}")

    model_config = AutoConfig.from_pretrained(str(model_config_path))
    if hasattr(model_config, "use_cache") and bool(model_config.use_cache):
        model_config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base",
        model_max_length=args.max_length,
        cache_dir=str(run_cache_dir),
        clean_up_tokenization_spaces=False,
    )
    tokenizer.clean_up_tokenization_spaces = False
    pad_idx = tokenizer.pad_token_id
    if pad_idx is None or int(pad_idx) < 0:
        fallback_pad = tokenizer.eos_token_id
        if fallback_pad is None or int(fallback_pad) < 0:
            fallback_pad = 0
        pad_idx = int(fallback_pad)
        tokenizer.pad_token_id = pad_idx
        log(f"Tokenizer | pad_token_id invalid, using {pad_idx}")

    tokenizer_vocab = len(tokenizer)
    if int(model_config.vocab_size) < tokenizer_vocab:
        old_vocab = int(model_config.vocab_size)
        model_config.vocab_size = tokenizer_vocab
        log(f"Model config | adjusted vocab_size from {old_vocab} to {tokenizer_vocab}")
    model_config.pad_token_id = int(tokenizer.pad_token_id)

    if args.total_batch_size % args.batch_size != 0:
        raise ValueError(
            f"total_batch_size ({args.total_batch_size}) must be divisible by batch_size ({args.batch_size})"
        )
    grad_accum = args.total_batch_size // args.batch_size

    train_data = datasets.load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        cache_dir=str(run_cache_dir),
    ).shuffle(seed=42)
    train_dataset = PreprocessedIterableDataset(
        train_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=args.workers)
    train_iter = iter(train_loader)

    def next_train_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    model = LlamaForCausalLM(model_config).to(device=device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = int(tokenizer.pad_token_id)

    optimizer, fused_adam = build_adam(
        model.parameters(),
        device,
        lr=args.lr,
        weight_decay=args.wd,
        betas=(float(args.beta), float(args.beta)),
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        min_lr_ratio=0.1,
    )
    log(f"Optimizer | type=Adam beta1=beta2={args.beta} wd={args.wd} fused={fused_adam} grad_accum={grad_accum}")

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "lr": [],
        "step_time_sec": [],
        "tokens_per_update": args.total_batch_size * args.max_length,
    }

    start_time = time.time()
    try:
        for step in range(1, args.steps + 1):
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            train_loss_accum = 0.0

            for _ in range(grad_accum):
                batch = next_train_batch()
                batch = {k: v.to(device) for k, v in batch.items()}
                batch["input_ids"] = sanitize_input_ids_for_vocab(
                    batch["input_ids"],
                    vocab_size=int(model.config.vocab_size),
                    unk_token_id=tokenizer.unk_token_id,
                )
                labels = batch["input_ids"].clone()
                labels[labels == tokenizer.pad_token_id] = -100
                with bf16_autocast(device, use_bf16):
                    outputs = model(**batch, labels=labels)
                    loss = outputs.loss / grad_accum
                loss.backward()
                train_loss_accum += float(loss.item()) * grad_accum

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step_time = time.time() - step_start
            if step == 1 or step % args.status_every == 0 or step == args.steps:
                log(
                    f"Progress | step={step}/{args.steps} train_loss={train_loss_accum:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} step_s={step_time:.2f}"
                )

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                val_loss, val_ppl, eval_tokens = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    eval_batch_size=args.eval_batch_size,
                    max_length=args.max_length,
                    target_eval_tokens=args.target_eval_tokens,
                    device=device,
                    use_bf16=use_bf16,
                    hf_cache_dir=run_cache_dir,
                )
                history["step"].append(step)
                history["train_loss"].append(train_loss_accum)
                history["val_loss"].append(val_loss)
                history["val_ppl"].append(val_ppl)
                history["lr"].append(float(scheduler.get_last_lr()[0]))
                history["step_time_sec"].append(step_time)
                log(
                    f"Eval | step={step}/{args.steps} train_loss={train_loss_accum:.4f} "
                    f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} eval_tokens={eval_tokens}"
                )
    except Exception as exc:
        log(f"Run failed | error={type(exc).__name__}: {exc}")
        raise

    elapsed = time.time() - start_time
    payload = {
        "args": {
            "beta": float(args.beta),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "total_batch_size": int(args.total_batch_size),
            "wd": float(args.wd),
            "steps": int(args.steps),
        },
        "history": history,
        "time_sec": float(elapsed),
        "meta": {
            "dataset": "C4",
            "dataset_hf_name": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "task": "next_token_prediction_causal",
            "model": "llama60m",
            "tokenizer": "t5-base",
            "dataset_mode": "hf_streaming",
            "hf_cache_dir": str(run_cache_dir),
        },
    }
    output_path = save_result(RESULT_PREFIX, args, payload)
    log(f"Saved to {output_path}")
    log(f"Timing summary | total_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
