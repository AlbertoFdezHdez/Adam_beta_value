import argparse
import os
import random
import shutil
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from beta_paths import (
    JOB_LOGS_ROOT,
    beta_tag,
    default_cache_dir,
    ensure_dir,
    language_payload,
    resolve_repo_path,
    save_result,
)
from packed_tokens import PackedTokens
from torch_accel import bf16_autocast, build_adamw, use_cuda_bf16
from init_batch_probe import build_probe_payload, write_probe_payload


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def estimate_loss(model, get_batch, device, use_bf16=False, eval_iters=50, split="val"):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x = get_batch(split)
        with bf16_autocast(device, use_bf16):
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                x[:, 1:].contiguous().view(-1),
            )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def safe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return None


def nearest_existing_path(path):
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def describe_disk_usage(path):
    base = nearest_existing_path(path)
    usage = shutil.disk_usage(base)
    return (
        f"disk total={format_bytes(usage.total)} "
        f"used={format_bytes(usage.used)} free={format_bytes(usage.free)} "
        f"checked_at={base}"
    )


def cached_file_status(path):
    path = Path(path)
    if not path.exists():
        return f"{path} (missing)"
    return f"{path} ({format_bytes(path.stat().st_size)})"


def require_local_dataset_dir(path: Path, dataset_name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} directory not found at {path}. "
            "Datasets must already exist under DATASETS_ROOT; this script does not download them."
        )
    return path


def build_log_path(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        "nanogpt_openwebtext",
        f"bs{int(args.batch_size)}",
        f"beta{beta_tag(args.beta)}",
        f"seed{int(args.seed)}",
    ]
    job_id = os.environ.get("SLURM_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job_id:
        parts.append(f"job{job_id}")
    if task_id:
        parts.append(f"task{task_id}")
    parts.append(stamp)
    return ensure_dir(JOB_LOGS_ROOT) / ("_".join(parts) + ".log")


class RunLogger:
    def __init__(self, path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.handle = self.path.open("a", encoding="utf-8")

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        tqdm.write(line)
        self.handle.write(line + "\n")
        self.handle.flush()

    def log_exception(self, prefix):
        self.log(prefix)
        for line in traceback.format_exc().rstrip().splitlines():
            self.log(line)

    def close(self):
        self.handle.close()

    def sample_batch(self, batch_size, device, generator=None):
        max_start = self.length - self.block_size - 1
        ix = torch.randint(0, max_start, (batch_size,), device=device, generator=generator)
        x = torch.empty((batch_size, self.block_size), dtype=torch.long, device=device)
        for i, s in enumerate(ix.tolist()):
            x[i] = torch.from_numpy(self.mem[s : s + self.block_size].astype(np.int64)).to(device)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--results_subdir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--target_tokens_per_update", type=int, default=262_144)
    parser.add_argument("--lr_max", type=float, default=6e-4)
    parser.add_argument("--min_lr_frac", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.001)
    parser.add_argument("--probe_init_batch_loss", action="store_true")
    parser.add_argument("--probe_output_subdir", type=str, default="init_batch_loss_probe")
    args = parser.parse_args()

    cache_dir = require_local_dataset_dir(
        resolve_repo_path(args.cache_dir, default_cache_dir("openwebtext")),
        "OpenWebText",
    )
    hf_cache_dir = resolve_repo_path(args.hf_cache_dir, default_cache_dir("huggingface"))
    log_path = build_log_path(args)
    logger = RunLogger(log_path)
    logger.log(f"Log file: {log_path}")
    logger.log(
        f"Environment | host={socket.gethostname()} | pid={os.getpid()} | python={sys.executable} | "
        f"cuda_available={torch.cuda.is_available()} | slurm_job_id={os.environ.get('SLURM_JOB_ID', '-') } | "
        f"slurm_array_task_id={os.environ.get('SLURM_ARRAY_TASK_ID', '-')}"
    )
    logger.log(f"Arguments | {vars(args)}")
    logger.log(f"Packed dir | cache_dir={cache_dir} | {describe_disk_usage(cache_dir)}")
    logger.log(f"HF cache path argument | hf_cache_dir={hf_cache_dir} | note=unused_in_training_only_mode")
    logger.log(
        "Packed file status | "
        f"train_bin={cached_file_status(cache_dir / 'owt_train_gpt2bpe_u16.bin')} | "
        f"train_len={cached_file_status(cache_dir / 'owt_train_gpt2bpe_u16.len.npy')} | "
        f"val_bin={cached_file_status(cache_dir / 'owt_val_gpt2bpe_u16.bin')} | "
        f"val_len={cached_file_status(cache_dir / 'owt_val_gpt2bpe_u16.len.npy')}"
    )

    try:
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = use_cuda_bf16(device)
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        logger.log(f"Device | device={device} | bf16={use_bf16}")

        logger.log("Model init start")
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=50304,
                n_positions=args.block_size,
                n_ctx=args.block_size,
                n_embd=384,
                n_layer=6,
                n_head=6,
                resid_pdrop=0.2,
                embd_pdrop=0.2,
                attn_pdrop=0.2,
            )
        ).to(device)
        logger.log(
            f"Model init done | params={sum(p.numel() for p in model.parameters())} | block_size={args.block_size}"
        )

        if args.val_fraction != 0.001:
            logger.log(f"val_fraction argument ignored in training-only mode | val_fraction={args.val_fraction}")

        train_pack = PackedTokens(cache_dir, "owt_train_gpt2bpe_u16", args.block_size, log=logger.log)
        val_pack = PackedTokens(cache_dir, "owt_val_gpt2bpe_u16", args.block_size, log=logger.log)
        logger.log(
            f"Packed datasets ready | train_tokens={train_pack.length} | val_tokens={val_pack.length} | packed_dir={cache_dir}"
        )

        def get_batch(split, generator=None):
            if split == "train":
                return train_pack.sample_batch(args.batch_size, device, generator=generator)
            return val_pack.sample_batch(args.batch_size, device, generator=generator)

        if args.probe_init_batch_loss:
            t_probe = time.time()
            g_probe = torch.Generator(device=device).manual_seed(args.seed)
            x = get_batch("train", generator=g_probe)
            model.train()
            with torch.no_grad():
                with bf16_autocast(device, use_bf16):
                    logits = model(input_ids=x).logits
                    loss = F.cross_entropy(
                        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                        x[:, 1:].contiguous().view(-1),
                    )
            payload = build_probe_payload(
                args=args,
                case_id="nanogpt_openwebtext",
                label="NanoGPT + OpenWebText",
                split="train",
                family="language",
                loss_value=float(loss.item()),
                batch={"input_ids": x},
                valid_targets=int(x[:, 1:].numel()),
                model=model,
                meta={
                    "dataset_root": str(cache_dir),
                    "hf_cache_dir": str(hf_cache_dir),
                    "block_size": int(args.block_size),
                    "tokenizer": "gpt2",
                },
                use_bf16=use_bf16,
                device=device,
                start_time=t_probe,
            )
            out_path = write_probe_payload(args, "nanogpt_openwebtext", payload)
            logger.log(f"Saved probe to {out_path}")
            return

        tokens_per_micro = args.batch_size * args.block_size
        grad_accum = max(1, args.target_tokens_per_update // max(1, tokens_per_micro))
        tokens_per_update = tokens_per_micro * grad_accum

        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and ("ln" not in n.lower()) and ("norm" not in n.lower()) and (not n.endswith(".bias")):
                decay.append(p)
            else:
                no_decay.append(p)

        optimizer, fused_adamw = build_adamw(
            [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
            device=device,
            lr=args.lr_max,
            betas=(args.beta, args.beta),
        )
        logger.log(
            f"Optimizer ready | beta={args.beta} | lr_max={args.lr_max} | wd={args.wd} | "
            f"grad_accum={grad_accum} | tokens_per_update={tokens_per_update} | fused={fused_adamw}"
        )

        min_lr = args.min_lr_frac * args.lr_max

        def lr_at(step):
            if step < args.warmup_steps:
                return args.lr_max * (step + 1) / max(1, args.warmup_steps)
            t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            return float(min_lr + 0.5 * (args.lr_max - min_lr) * (1 + np.cos(np.pi * t)))

        history = {
            "iter": [],
            "train_loss": [],
            "val_loss": [],
            "train_ppl": [],
            "val_ppl": [],
            "lr": [],
            "tokens_per_update": int(tokens_per_update),
            "grad_accum": int(grad_accum),
            "block_size": int(args.block_size),
            "iter_max": int(args.steps),
            "warmup_steps": int(args.warmup_steps),
            "eval_interval": int(args.eval_interval),
            "eval_iters": int(args.eval_iters),
            "target_tokens_per_update": int(args.target_tokens_per_update),
        }

        g_val = torch.Generator(device=device).manual_seed(123456789)
        g_train_eval = torch.Generator(device=device).manual_seed(987654321)

        t_start = time.time()
        train_step_sec_total = 0.0
        logger.log(
            f"Run config | steps={args.steps} | eval_interval={args.eval_interval} | "
            f"eval_iters={args.eval_iters} | grad_accum={grad_accum} | "
            f"tokens_per_update={tokens_per_update}"
        )
        pbar = tqdm(range(1, args.steps + 1), leave=True, dynamic_ncols=True)
        last_val = None

        model.train()
        for it in pbar:
            step_t0 = time.time()
            lr = lr_at(it - 1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            for _ in range(grad_accum):
                x = get_batch("train")
                with bf16_autocast(device, use_bf16):
                    logits = model(input_ids=x).logits
                    loss = F.cross_entropy(
                        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                        x[:, 1:].contiguous().view(-1),
                    )
                (loss / grad_accum).backward()
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            train_step_sec_total += time.time() - step_t0

            train_step_loss = total_loss / grad_accum
            postfix = {
                "loss": f"{train_step_loss:.4f}",
                "lr": f"{lr:.2e}",
                "tpu": f"{tokens_per_update / 1e6:.2f}M",
                "step_s": f"{train_step_sec_total / it:.2f}",
            }
            if last_val is not None:
                postfix["val"] = f"{last_val:.4f}"
            pbar.set_postfix(postfix)

            should_eval = (it == 1) or (it % args.eval_interval == 0) or (it == args.steps)
            if should_eval:
                eval_t0 = time.time()
                train_eval = estimate_loss(
                    model,
                    lambda s: get_batch(s, generator=g_train_eval),
                    device=device,
                    use_bf16=use_bf16,
                    eval_iters=args.eval_iters,
                    split="train",
                )
                val_loss = estimate_loss(
                    model,
                    lambda s: get_batch(s, generator=g_val),
                    device=device,
                    use_bf16=use_bf16,
                    eval_iters=args.eval_iters,
                    split="val",
                )
                last_val = val_loss

                history["iter"].append(int(it))
                history["lr"].append(float(lr))
                history["train_loss"].append(float(train_eval))
                history["val_loss"].append(float(val_loss))
                history["train_ppl"].append(float(np.exp(min(20.0, train_eval))))
                history["val_ppl"].append(float(np.exp(min(20.0, val_loss))))
                eval_sec = time.time() - eval_t0

                logger.log(
                    f"iter {it:06d}/{args.steps} | "
                    f"train {train_eval:.4f} ppl {np.exp(min(20.0, train_eval)):.2f} | "
                    f"val {val_loss:.4f} ppl {np.exp(min(20.0, val_loss)):.2f} | "
                    f"time step_avg {train_step_sec_total / it:.2f}s eval={eval_sec:.1f}s total={time.time() - t_start:.1f}s | "
                    f"accum {grad_accum} (tpu {tokens_per_update / 1e6:.2f}M) | "
                    f"wd {args.wd}"
                )

        total_sec = time.time() - t_start
        out_path = save_result(
            "nanogpt_openwebtext",
            args,
            language_payload(args, history, total_sec, "openwebtext"),
        )
        logger.log(f"Saved to {out_path}")
        logger.log(
            f"Timing summary | total={total_sec:.1f}s | avg_step={train_step_sec_total / max(1, args.steps):.2f}s"
        )
    except Exception:
        logger.log_exception(
            f"Run failed | cache_dir={cache_dir} | hf_cache_dir={hf_cache_dir} | {describe_disk_usage(cache_dir)}"
        )
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
