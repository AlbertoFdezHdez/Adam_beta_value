from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder

from beta_paths import default_dataset_dir, resolve_repo_path


def _has_partitioned_layout(root: Path) -> bool:
    train_parts = [root / f"train.X{i}" for i in range(1, 5)]
    return all(part.exists() for part in train_parts) and (root / "val.X").exists()


def _has_classic_layout(root: Path) -> bool:
    train_dir = root / "train"
    val_dir = root / "val"
    validation_dir = root / "validation"
    return train_dir.exists() and (val_dir.exists() or validation_dir.exists())


def _version_sort_key(path: Path):
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def _candidate_roots(requested_root: Path) -> list[Path]:
    candidates = [requested_root]
    if requested_root.name != "imagenet100":
        candidates.append(requested_root / "imagenet100")

    expanded: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        if candidate in seen:
            continue
        expanded.append(candidate)
        seen.add(candidate)

        versions_dir = candidate / "versions"
        if not versions_dir.is_dir():
            continue

        complete_versions = []
        for marker in candidate.glob("*.complete"):
            if marker.is_file():
                version_dir = versions_dir / marker.stem
                if version_dir.is_dir():
                    complete_versions.append(version_dir)

        for version_dir in sorted(complete_versions, key=_version_sort_key):
            if version_dir not in seen:
                expanded.append(version_dir)
                seen.add(version_dir)

        for version_dir in sorted((path for path in versions_dir.iterdir() if path.is_dir()), key=_version_sort_key):
            if version_dir not in seen:
                expanded.append(version_dir)
                seen.add(version_dir)

    return expanded


def resolve_imagenet100_root(raw_path: str | None) -> Path:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("imagenet100"))
    candidates = _candidate_roots(requested_root)

    for candidate in candidates:
        if _has_partitioned_layout(candidate) or _has_classic_layout(candidate):
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "ImageNet100 not found in any supported layout. Checked: "
        f"{checked}"
    )


def _target_remapper(remap: dict[int, int]):
    return lambda target, remap=remap: remap[target]


def _build_partitioned_datasets(root: Path, train_transform, val_transform):
    val_dataset = ImageFolder(root / "val.X", val_transform)
    global_class_to_idx = dict(val_dataset.class_to_idx)
    global_class_names = set(global_class_to_idx)

    train_parts = []
    seen_train_classes = set()
    for part_idx in range(1, 5):
        part = ImageFolder(root / f"train.X{part_idx}", train_transform)
        unknown_classes = sorted(set(part.class_to_idx) - global_class_names)
        if unknown_classes:
            raise ValueError(
                f"ImageNet100 split train.X{part_idx} has classes missing in val.X: {unknown_classes}"
            )

        remap = {
            local_idx: global_class_to_idx[class_name]
            for class_name, local_idx in part.class_to_idx.items()
        }
        part.target_transform = _target_remapper(remap)
        train_parts.append(part)
        seen_train_classes.update(part.class_to_idx)

    missing_train_classes = sorted(global_class_names - seen_train_classes)
    if missing_train_classes:
        raise ValueError(
            "ImageNet100 training splits do not cover all validation classes. "
            f"Missing: {missing_train_classes}"
        )

    train_dataset = ConcatDataset(train_parts)
    num_classes = len(global_class_to_idx)
    return train_dataset, val_dataset, num_classes


def build_imagenet100_datasets(raw_path: str | None, train_transform, val_transform):
    root = resolve_imagenet100_root(raw_path)

    if _has_partitioned_layout(root):
        train_dataset, val_dataset, num_classes = _build_partitioned_datasets(
            root,
            train_transform,
            val_transform,
        )
        return root, train_dataset, val_dataset, num_classes

    train_dir = root / "train"
    val_dir = root / "val"
    if not val_dir.exists() and (root / "validation").exists():
        val_dir = root / "validation"

    train_dataset = ImageFolder(train_dir, train_transform)
    val_dataset = ImageFolder(val_dir, val_transform)
    num_classes = len(train_dataset.classes)
    return root, train_dataset, val_dataset, num_classes
