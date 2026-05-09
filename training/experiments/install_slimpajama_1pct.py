from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from beta_paths import default_dataset_dir, resolve_repo_path


REPO_ID = "cerebras/SlimPajama-627B"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install a deterministic 1% local subset of SlimPajama.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include_validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo_id", type=str, default=REPO_ID)
    return parser.parse_args()


def allocate_counts_by_chunk(grouped: dict[str, list[str]], fraction: float, seed: int) -> list[str]:
    all_files = sorted(file for files in grouped.values() for file in files)
    target_total = max(1, math.ceil(len(all_files) * fraction))
    rng = random.Random(seed)

    raw_targets = []
    for chunk_name, files in sorted(grouped.items()):
        exact = len(files) * fraction
        base = min(len(files), int(math.floor(exact)))
        raw_targets.append([chunk_name, files, exact, base, exact - base])

    selected = sum(item[3] for item in raw_targets)
    remainder = max(0, target_total - selected)
    raw_targets.sort(key=lambda item: (-item[4], item[0]))
    idx = 0
    while remainder > 0 and raw_targets:
        chunk_name, files, exact, base, frac = raw_targets[idx]
        if base < len(files):
            raw_targets[idx][3] += 1
            remainder -= 1
        idx = (idx + 1) % len(raw_targets)
        if idx == 0 and all(item[3] >= len(item[1]) for item in raw_targets):
            break

    selected_files: list[str] = []
    for chunk_name, files, _exact, count, _frac in sorted(raw_targets, key=lambda item: item[0]):
        if count <= 0:
            continue
        chosen = rng.sample(sorted(files), k=count) if count < len(files) else sorted(files)
        selected_files.extend(sorted(chosen))
    return selected_files


def main() -> None:
    args = build_args()
    output_dir = resolve_repo_path(args.output_dir, default_dataset_dir("slimpajama"))
    manifest_path = output_dir / "slimpajama_1pct_manifest.json"

    if output_dir.exists() and args.overwrite:
        log(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    log(f"Listing repo files for dataset: {args.repo_id}")
    repo_files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    log(f"Repo file count: {len(repo_files)}")

    train_files = sorted(
        file_path
        for file_path in repo_files
        if file_path.startswith("train/") and file_path.endswith(".jsonl.zst")
    )
    validation_files = sorted(
        file_path
        for file_path in repo_files
        if (file_path.startswith("validation/") or file_path.startswith("val/")) and file_path.endswith(".jsonl.zst")
    )
    readme_files = [file_path for file_path in repo_files if file_path == "README.md"]

    if not train_files:
        raise RuntimeError("No train .jsonl.zst files found in SlimPajama repo listing.")

    grouped: dict[str, list[str]] = defaultdict(list)
    for file_path in train_files:
        parts = file_path.split("/")
        chunk_name = parts[1] if len(parts) > 2 else "train"
        grouped[chunk_name].append(file_path)

    selected_train_files = allocate_counts_by_chunk(grouped, args.fraction, args.seed)
    if not selected_train_files:
        raise RuntimeError("Selected train file list is empty.")

    allow_patterns = list(readme_files)
    allow_patterns.extend(selected_train_files)
    if args.include_validation:
        allow_patterns.extend(validation_files)

    log(
        f"Downloading SlimPajama subset | train_files_total={len(train_files)} "
        f"train_files_selected={len(selected_train_files)} validation_files={len(validation_files)} "
        f"target_dir={output_dir}"
    )
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
    )

    manifest = {
        "repo_id": args.repo_id,
        "fraction": args.fraction,
        "seed": args.seed,
        "selection_mode": "deterministic_random_by_chunk",
        "train_files_total": len(train_files),
        "train_files_selected": len(selected_train_files),
        "validation_files_selected": len(validation_files) if args.include_validation else 0,
        "selected_train_files": selected_train_files,
        "selected_validation_files": validation_files if args.include_validation else [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Saved manifest to {manifest_path}")
    log("SlimPajama 1% installation finished")


if __name__ == "__main__":
    main()
