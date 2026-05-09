from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from beta_paths import RESULTS_ROOT, ensure_dir
from torch_accel import bf16_autocast


def probe_output_path(args: Any, prefix: str) -> Path:
    subdir = getattr(args, "probe_output_subdir", "init_batch_loss_probe")
    output_dir = ensure_dir(RESULTS_ROOT / subdir)
    return output_dir / f"{prefix}_seed{int(args.seed)}.json"


def write_probe_payload(args: Any, prefix: str, payload: dict[str, Any]) -> Path:
    out_path = probe_output_path(args, prefix)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def tensor_shapes(batch: Any) -> dict[str, list[int]]:
    if isinstance(batch, dict):
        return {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")}
    if isinstance(batch, (tuple, list)):
        out: dict[str, list[int]] = {}
        for idx, value in enumerate(batch):
            if hasattr(value, "shape"):
                out[f"item_{idx}"] = list(value.shape)
        return out
    return {}


def count_valid_targets(labels: torch.Tensor) -> int:
    if (labels == -100).any():
        return int((labels != -100).sum().item())
    return int(labels.numel())


def build_probe_payload(
    *,
    args: Any,
    case_id: str,
    label: str,
    split: str,
    family: str,
    loss_value: float,
    batch: Any,
    valid_targets: int,
    model: torch.nn.Module,
    meta: dict[str, Any] | None = None,
    use_bf16: bool,
    device: torch.device,
    start_time: float,
) -> dict[str, Any]:
    return {
        "case": {
            "case_id": case_id,
            "label": label,
            "study_split": split,
            "family": family,
            "batch_size": int(args.batch_size),
        },
        "seed": int(args.seed),
        "device": str(device),
        "use_bf16": bool(use_bf16),
        "probe_kind": "single_train_batch_forward_only",
        "optimizer_used": False,
        "backward_used": False,
        "model_train_mode": True,
        "init_batch_loss": float(loss_value),
        "time_sec": float(time.time() - start_time),
        "meta": {
            "batch_shapes": tensor_shapes(batch),
            "valid_targets": int(valid_targets),
            "model_params": int(sum(p.numel() for p in model.parameters())),
            **(meta or {}),
        },
    }


def run_classification_probe(
    *,
    args: Any,
    prefix: str,
    case_id: str,
    label: str,
    split: str,
    family: str,
    train_loader,
    model: torch.nn.Module,
    device: torch.device,
    use_bf16: bool,
    loss_fn,
    meta: dict[str, Any] | None = None,
) -> Path:
    t0 = time.time()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    model.train()
    with torch.no_grad():
        with bf16_autocast(device, use_bf16):
            logits = model(x)
            loss = loss_fn(logits, y)
    payload = build_probe_payload(
        args=args,
        case_id=case_id,
        label=label,
        split=split,
        family=family,
        loss_value=float(loss.item()),
        batch=(x, y),
        valid_targets=int(y.numel()),
        model=model,
        meta=meta,
        use_bf16=use_bf16,
        device=device,
        start_time=t0,
    )
    return write_probe_payload(args, prefix, payload)
