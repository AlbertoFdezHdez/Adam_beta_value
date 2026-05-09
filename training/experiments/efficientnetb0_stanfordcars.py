from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from beta_paths import classification_payload, save_result
from stanfordcars_data import build_stanfordcars_datasets
from torch_accel import bf16_autocast, build_adam, use_cuda_bf16
from init_batch_probe import run_classification_probe


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--results_subdir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--probe_init_batch_loss", action="store_true")
    parser.add_argument("--probe_output_subdir", type=str, default="init_batch_loss_probe")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = use_cuda_bf16(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    tf_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.5, 2.0), value="random"),
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

    data_root, train_set, val_set, num_classes = build_stanfordcars_datasets(args.data_root, tf_train, tf_val)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)

    if args.probe_init_batch_loss:
        out_path = run_classification_probe(
            args=args,
            prefix="efficientnetb0_stanfordcars",
            case_id="efficientnetb0_stanfordcars",
            label="EfficientNet-B0 + Stanford Cars",
            split="val",
            family="vision",
            train_loader=train_loader,
            model=model,
            device=device,
            use_bf16=use_bf16,
            loss_fn=lambda logits, y: F.cross_entropy(logits, y, label_smoothing=args.label_smoothing),
            meta={
                "dataset_root": str(data_root),
                "num_classes": int(num_classes),
            },
        )
        print(f"Saved probe to {out_path}")
        return

    optimizer, fused_adam = build_adam(
        model.parameters(),
        device=device,
        lr=args.lr,
        weight_decay=args.wd,
        betas=(args.beta, args.beta),
    )

    epochs = int(args.epochs)
    max_train_batches = len(train_loader) if int(args.max_train_batches) <= 0 else min(len(train_loader), int(args.max_train_batches))
    max_val_batches = len(val_loader) if int(args.max_val_batches) <= 0 else min(len(val_loader), int(args.max_val_batches))
    steps_per_epoch = max_train_batches
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(100, int(0.05 * total_steps))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    global_step = 0
    t_start = time.time()
    print("Stanford Cars root:", data_root)
    print("Num classes:", num_classes)
    print("Precision mode:", "bf16_autocast" if use_bf16 else "fp32")
    print("Adam fused:", fused_adam)
    print(
        f"Run config | epochs={epochs} | train_batches={len(train_loader)} | "
        f"val_batches={len(val_loader)} | effective_train_batches={max_train_batches} | "
        f"effective_val_batches={max_val_batches} | wd={args.wd} | workers={args.num_workers}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        train_batches = 0
        train_t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=True, dynamic_ncols=True)
        for x, y in pbar:
            if train_batches >= max_train_batches:
                break
            x, y = x.to(device), y.to(device)

            if global_step < warmup_steps:
                lr = args.lr * (global_step + 1) / max(1, warmup_steps)
            else:
                t = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr = float(args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * t)))

            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with bf16_autocast(device, use_bf16):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            acc = accuracy(logits, y)
            running_loss += loss.item()
            running_acc += acc
            train_batches += 1
            global_step += 1

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
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        val_t0 = time.time()
        with torch.no_grad():
            for x, y in val_loader:
                if val_batches >= max_val_batches:
                    break
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

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f} | "
            f"time train={train_sec:.1f}s val={val_sec:.1f}s"
        )

    payload = classification_payload(args, history, time.time() - t_start)
    payload["meta"] = {
        "dataset": "stanfordcars",
        "task": "image_classification",
        "model": "efficientnet_b0",
        "optimizer": "adam",
        "weight_decay": float(args.wd),
        "lr": float(args.lr),
        "min_lr": float(args.min_lr),
        "epochs": int(args.epochs),
        "num_classes": int(num_classes),
        "data_root": str(data_root),
    }
    out_path = save_result("efficientnetb0_stanfordcars", args, payload)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
