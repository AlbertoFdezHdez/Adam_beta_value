from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

from beta_paths import JOB_LOGS_ROOT, default_cache_dir, ensure_dir, resolve_repo_path
from hf_dataset_local import load_local_dataset_dict
from packed_tokens import safe_len, write_packed_split


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        ensure_dir(log_path.parent)
        self.handle = log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def build_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    parts = ["preprocess_openwebtext_gpt2"]
    if job_id:
        parts.append(f"job{job_id}")
    if task_id:
        parts.append(f"task{task_id}")
    parts.append(stamp)
    return ensure_dir(JOB_LOGS_ROOT) / ("_".join(parts) + ".log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--packed_dir", type=str, default=None)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--batch_docs", type=int, default=1024)
    parser.add_argument("--val_fraction", type=float, default=0.001)
    parser.add_argument("--split_seed", type=int, default=123456)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_dir = resolve_repo_path(args.dataset_dir, default_cache_dir("openwebtext"))
    packed_dir = resolve_repo_path(args.packed_dir, dataset_dir)
    hf_cache_dir = resolve_repo_path(args.hf_cache_dir, default_cache_dir("huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    log_path = build_log_path()
    logger = RunLogger(log_path)
    t0 = time.time()

    try:
        logger.log(f"Log file: {log_path}")
        logger.log(
            f"Environment | host={socket.gethostname()} | pid={os.getpid()} | python={sys.executable} "
            f"| tokenizer_parallelism={os.environ.get('TOKENIZERS_PARALLELISM', 'unset')}"
        )
        logger.log(
            f"Arguments | dataset_dir={dataset_dir} | packed_dir={packed_dir} | hf_cache_dir={hf_cache_dir} "
            f"| tokenizer_name={args.tokenizer_name} | batch_docs={args.batch_docs} | "
            f"val_fraction={args.val_fraction} | split_seed={args.split_seed} | overwrite={args.overwrite}"
        )

        logger.log("Tokenizer load start")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=str(hf_cache_dir))
        tokenizer.model_max_length = int(1e30)
        logger.log(
            f"Tokenizer load done | name={args.tokenizer_name} | fast={getattr(tokenizer, 'is_fast', False)} "
            f"| vocab_size={getattr(tokenizer, 'vocab_size', 'unknown')}"
        )

        logger.log("Dataset load start")
        dataset_dict, source_mode = load_local_dataset_dict(dataset_dir, "openwebtext")
        logger.log(
            f"Dataset load done | source={source_mode} | splits={list(dataset_dict.keys())} | "
            f"train_rows={safe_len(dataset_dict['train']) if 'train' in dataset_dict else 'missing'}"
        )

        if "validation" in dataset_dict:
            train_split = dataset_dict["train"]
            val_split = dataset_dict["validation"]
            logger.log(
                f"Dataset split reuse | train_rows={safe_len(train_split)} | val_rows={safe_len(val_split)}"
            )
        else:
            logger.log(
                f"Dataset split start | source=train | val_fraction={args.val_fraction} | split_seed={args.split_seed}"
            )
            split = dataset_dict["train"].train_test_split(test_size=args.val_fraction, seed=args.split_seed)
            train_split = split["train"]
            val_split = split["test"]
            logger.log(
                f"Dataset split done | train_rows={safe_len(train_split)} | val_rows={safe_len(val_split)}"
            )

        train_stats = write_packed_split(
            train_split,
            tokenizer,
            packed_dir,
            "owt_train_gpt2bpe_u16",
            batch_docs=args.batch_docs,
            overwrite=args.overwrite,
            log=logger.log,
        )
        val_stats = write_packed_split(
            val_split,
            tokenizer,
            packed_dir,
            "owt_val_gpt2bpe_u16",
            batch_docs=args.batch_docs,
            overwrite=args.overwrite,
            log=logger.log,
        )

        summary = {
            "dataset_dir": str(dataset_dir),
            "packed_dir": str(packed_dir),
            "tokenizer_name": args.tokenizer_name,
            "batch_docs": args.batch_docs,
            "train": train_stats,
            "val": val_stats,
            "elapsed_sec_total": time.time() - t0,
        }
        logger.log("Summary | " + json.dumps(summary, ensure_ascii=True))
        logger.log(f"Preprocess completed | elapsed={format_seconds(summary['elapsed_sec_total'])}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
