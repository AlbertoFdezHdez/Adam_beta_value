from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from beta_paths import classification_payload, default_dataset_dir, resolve_repo_path, save_result
from torch_accel import bf16_autocast, build_adamw, use_cuda_bf16
from init_batch_probe import run_classification_probe


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


def is_food101_dir(path: Path) -> bool:
    return (path / "meta").exists() and (path / "images").exists()


def resolve_food101_roots(raw_path: str | None) -> tuple[Path, Path]:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("food101"))
    candidates = [requested_root]
    if requested_root.name != "food-101":
        candidates.append(requested_root / "food-101")

    for candidate in candidates:
        if is_food101_dir(candidate):
            return candidate, candidate.parent

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Food-101 not found. Expected the extracted dataset under one of: "
        f"{checked}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--results_subdir", type=str, default=None)
    parser.add_argument("--probe_init_batch_loss", action="store_true")
    parser.add_argument("--probe_output_subdir", type=str, default="init_batch_loss_probe")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = use_cuda_bf16(device)

    dataset_dir, torchvision_root = resolve_food101_roots(args.data_root)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    tf_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    tf_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = datasets.Food101(root=str(torchvision_root), split="train", download=False, transform=tf_train)
    val_set = datasets.Food101(root=str(torchvision_root), split="test", download=False, transform=tf_val)
    num_classes = len(train_set.classes)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model.to(device)

    if args.probe_init_batch_loss:
        out_path = run_classification_probe(
            args=args,
            prefix="resnet50_food101",
            case_id="resnet50_food101",
            label="ResNet50 + Food101",
            split="train",
            family="vision",
            train_loader=train_loader,
            model=model,
            device=device,
            use_bf16=use_bf16,
            loss_fn=lambda logits, y: F.cross_entropy(logits, y, label_smoothing=0.1),
            meta={
                "dataset_root": str(dataset_dir),
                "torchvision_root": str(torchvision_root),
                "num_classes": int(num_classes),
            },
        )
        print(f"Saved probe to {out_path}")
        return

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "bn" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer, fused_adamw = build_adamw(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        device=device,
        lr=5e-4,
        betas=(args.beta, args.beta),
    )

    epochs = int(args.epochs)
    steps_per_epoch = len(train_loader)
    max_val_batches = len(val_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    min_lr = 3e-5
    base_lr = 3e-4

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    global_step = 0
    t_start = time.time()

    print("Food-101 dataset dir:", dataset_dir)
    print("Food-101 torchvision root:", torchvision_root)
    print("Precision mode:", "bf16_autocast" if use_bf16 else "fp32")
    print("AdamW fused:", fused_adamw)
    print("Train classes:", num_classes)
    print("Val classes:", len(val_set.classes))
    print(
        f"Run config | epochs={epochs} | "
        f"train_batches={steps_per_epoch} | "
        f"val_batches={max_val_batches}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        train_batches = 0
        train_t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=True, dynamic_ncols=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            if global_step < warmup_steps:
                lr = base_lr * (global_step + 1) / max(1, warmup_steps)
            else:
                t = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with bf16_autocast(device, use_bf16):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss.backward()
            optimizer.step()

            acc = accuracy(logits, y)
            running_loss += loss.item()
            running_acc += acc
            global_step += 1
            train_batches += 1

            pbar.set_postfix(
                loss=f"{running_loss / train_batches:.4f}",
                acc=f"{running_acc / train_batches:.4f}",
                lr=f"{lr:.2e}",
            )

        train_loss = running_loss / train_batches
        train_acc = running_acc / train_batches
        pbar.close()
        train_sec = time.time() - train_t0

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        val_batches = 0
        val_t0 = time.time()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with bf16_autocast(device, use_bf16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                val_loss += loss.item()
                val_acc += accuracy(logits, y)
                val_batches += 1

        val_loss /= val_batches
        val_acc /= val_batches
        val_sec = time.time() - val_t0
        epoch_sec = train_sec + val_sec

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f} | "
            f"batches train={train_batches} val={val_batches} | "
            f"time train={train_sec:.1f}s val={val_sec:.1f}s epoch={epoch_sec:.1f}s"
        )

    out_path = save_result(
        "resnet50_food101",
        args,
        classification_payload(args, history, time.time() - t_start),
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
