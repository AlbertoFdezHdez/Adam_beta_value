from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any


EXPERIMENTS_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENTS_ROOT.parent
REPO_ROOT = CODE_ROOT.parent

DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", "/scratch1/hernanal/datasets")).expanduser()
DESCARGAS_ROOT = REPO_ROOT / "descargas"
RESULTS_ROOT = DESCARGAS_ROOT / "results"
JOB_LOGS_ROOT = DESCARGAS_ROOT / "logs" / "jobs"

DATASET_DIR_NAMES = {
    "bookcorpus": "bookcorpus",
    "c4": "c4",
    "caltech256": "caltech-256",
    "cifar100": "cifar-100-python",
    "food101": "food-101",
    "imagenet100": "imagenet100",
    "tinyimagenet": "tiny-imagenet-200",
    "openwebtext": "openwebtext",
    "stanfordcars": "stanford-cars",
    "slimpajama": "slimpajama",
    "wikitext": "wikitext",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(raw_path: str | Path | None, default: Path) -> Path:
    if raw_path in (None, ""):
        return default
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def default_dataset_dir(name: str) -> Path:
    return DATASETS_ROOT / DATASET_DIR_NAMES.get(name, name)


def default_cache_dir(name: str) -> Path:
    if name == "huggingface":
        return DATASETS_ROOT
    return default_dataset_dir(name)


def beta_tag(beta: float) -> str:
    return f"{beta:.5f}".rstrip("0").rstrip(".")


def result_filename(prefix: str, batch_size: int, beta: float, seed: int) -> str:
    return f"{prefix}_bs{batch_size}_beta{beta_tag(beta)}_seed{seed}.pkl"


def resolve_results_dir(subdir: str | Path | None = None) -> Path:
    if subdir in (None, ""):
        return RESULTS_ROOT

    path = Path(subdir)
    if path.is_absolute():
        raise ValueError("results_subdir must be relative to descargas/results")

    return RESULTS_ROOT / path


def core_args(args: Any) -> dict[str, Any]:
    return {
        "beta": float(args.beta),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
    }


def classification_payload(args: Any, history: dict[str, Any], time_sec: float) -> dict[str, Any]:
    return {
        "args": core_args(args),
        "history": history,
        "time_sec": float(time_sec),
    }


def language_payload(
    args: Any,
    history: dict[str, Any],
    time_sec: float,
    dataset_name: str,
) -> dict[str, Any]:
    return {
        "args": core_args(args),
        "history": history,
        "time_sec": float(time_sec),
        "meta": {
            "dataset": dataset_name,
            "task": "next_token_prediction_causal",
            "tokenizer": "gpt2_bpe",
        },
    }


def save_result(prefix: str, args: Any, payload: dict[str, Any]) -> Path:
    output_dir = ensure_dir(resolve_results_dir(getattr(args, "results_subdir", None)))
    out_path = output_dir / result_filename(prefix, args.batch_size, args.beta, args.seed)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    return out_path
