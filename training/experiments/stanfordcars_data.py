from __future__ import annotations

from pathlib import Path

from torchvision.datasets import ImageFolder

from beta_paths import default_dataset_dir, resolve_repo_path


def is_stanfordcars_root(path: Path) -> bool:
    return (path / "train").is_dir() and (path / "test").is_dir()


def resolve_stanfordcars_root(raw_path: str | None) -> Path:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("stanfordcars"))
    candidates = [requested_root]
    if requested_root.name != "stanford-cars":
        candidates.append(requested_root / "stanford-cars")

    for candidate in candidates:
        if is_stanfordcars_root(candidate):
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Stanford Cars not found. Expected the extracted dataset under one of: "
        f"{checked}"
    )


def build_stanfordcars_datasets(raw_path: str | None, train_transform, val_transform):
    root = resolve_stanfordcars_root(raw_path)
    train_dataset = ImageFolder(root / "train", transform=train_transform)
    val_dataset = ImageFolder(root / "test", transform=val_transform)
    num_classes = len(train_dataset.classes)
    return root, train_dataset, val_dataset, num_classes
