from __future__ import annotations

import inspect
from contextlib import nullcontext

import torch


def use_cuda_bf16(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(checker):
        return bool(checker())
    return False


def bf16_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def adamw_supports_fused() -> bool:
    return "fused" in inspect.signature(torch.optim.AdamW).parameters


def build_adamw(params, device: torch.device, **kwargs):
    if device.type == "cuda" and adamw_supports_fused():
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs), True
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(params, **kwargs), False


def adam_supports_fused() -> bool:
    return "fused" in inspect.signature(torch.optim.Adam).parameters


def build_adam(params, device: torch.device, **kwargs):
    if device.type == "cuda" and adam_supports_fused():
        try:
            return torch.optim.Adam(params, fused=True, **kwargs), True
        except (RuntimeError, TypeError):
            pass
    return torch.optim.Adam(params, **kwargs), False
