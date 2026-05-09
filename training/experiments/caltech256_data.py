from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from beta_paths import default_dataset_dir, resolve_repo_path
from hf_dataset_local import load_local_dataset_dict


HF_DATASET_NAME = "ilee0022/Caltech-256"


def _resolve_categories_dir(raw_path: str | None) -> Path:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("caltech256"))
    candidates = [requested_root]
    if requested_root.name != "caltech-256":
        candidates.append(requested_root / "caltech-256")

    expanded: list[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        expanded.append(candidate / "256_ObjectCategories")

    for candidate in expanded:
        if candidate.is_dir() and (candidate.name == "256_ObjectCategories" or (candidate / "256_ObjectCategories").is_dir()):
            if candidate.name == "256_ObjectCategories":
                return candidate
            return candidate / "256_ObjectCategories"

    checked = ", ".join(str(path) for path in expanded)
    raise FileNotFoundError(
        "Caltech-256 not found. Expected the extracted dataset under one of: "
        f"{checked}"
    )


class PathImageDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform
        self.targets = [target for _, target in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class HFImageDataset(Dataset):
    def __init__(self, dataset, class_to_idx: dict[str, int], transform=None):
        self.dataset = dataset
        self.class_to_idx = class_to_idx
        self.transform = transform
        text_column = dataset["text"] if "text" in dataset.column_names else []
        self.targets = [class_to_idx[str(name)] for name in text_column]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = self.class_to_idx[str(example["text"])]
        return image, target


def _resolve_requested_root(raw_path: str | None) -> Path:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("caltech256"))
    candidates = [requested_root]
    if requested_root.name != "caltech-256":
        candidates.append(requested_root / "caltech-256")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return requested_root


def build_caltech256_datasets(
    raw_path: str | None,
    train_transform,
    val_transform,
    *,
    split_seed: int = 123,
    train_per_class: int = 60,
    exclude_clutter: bool = True,
):
    requested_root = _resolve_requested_root(raw_path)

    if (requested_root / "dataset_dict.json").exists():
        dataset_dict, source_mode = load_local_dataset_dict(requested_root, HF_DATASET_NAME)
        if "train" not in dataset_dict:
            raise ValueError(f"Expected a train split in saved dataset at {requested_root}")
        val_split_name = "validation" if "validation" in dataset_dict else ("test" if "test" in dataset_dict else None)
        if val_split_name is None:
            raise ValueError(f"Expected a validation or test split in saved dataset at {requested_root}")

        train_split = dataset_dict["train"]
        val_split = dataset_dict[val_split_name]

        if exclude_clutter:
            train_split = train_split.filter(lambda example: str(example["text"]).strip().lower() != "clutter")
            val_split = val_split.filter(lambda example: str(example["text"]).strip().lower() != "clutter")

        class_names = sorted({str(example["text"]) for example in train_split})
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        train_dataset = HFImageDataset(train_split, class_to_idx=class_to_idx, transform=train_transform)
        val_dataset = HFImageDataset(val_split, class_to_idx=class_to_idx, transform=val_transform)
        num_classes = len(class_names)

        metadata = {
            "categories_dir": requested_root,
            "num_classes": num_classes,
            "train_per_class": None,
            "exclude_clutter": exclude_clutter,
            "split_seed": split_seed,
            "class_names": class_names,
            "source_mode": source_mode,
            "val_split_name": val_split_name,
        }
        return requested_root, train_dataset, val_dataset, num_classes, metadata

    categories_dir = _resolve_categories_dir(raw_path)
    full_dataset = ImageFolder(categories_dir)

    valid_class_names = []
    valid_old_targets = []
    for old_target, class_name in enumerate(full_dataset.classes):
        if exclude_clutter and class_name.lower().startswith("257.clutter"):
            continue
        valid_class_names.append(class_name)
        valid_old_targets.append(old_target)

    old_to_new = {old_target: new_target for new_target, old_target in enumerate(valid_old_targets)}
    grouped: dict[int, list[str]] = {old_target: [] for old_target in valid_old_targets}
    for path, old_target in full_dataset.samples:
        if old_target in grouped:
            grouped[old_target].append(path)

    rng = random.Random(split_seed)
    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []

    for old_target in valid_old_targets:
        paths = sorted(grouped[old_target])
        if len(paths) < 2:
            raise ValueError(f"Caltech-256 class has fewer than 2 images: target={old_target}")

        rng.shuffle(paths)
        if len(paths) <= train_per_class:
            train_count = max(1, int(0.8 * len(paths)))
        else:
            train_count = train_per_class
        train_count = min(train_count, len(paths) - 1)
        new_target = old_to_new[old_target]
        train_samples.extend((path, new_target) for path in paths[:train_count])
        val_samples.extend((path, new_target) for path in paths[train_count:])

    train_dataset = PathImageDataset(train_samples, transform=train_transform)
    val_dataset = PathImageDataset(val_samples, transform=val_transform)
    num_classes = len(valid_class_names)

    metadata = {
        "categories_dir": categories_dir,
        "num_classes": num_classes,
        "train_per_class": train_per_class,
        "exclude_clutter": exclude_clutter,
        "split_seed": split_seed,
        "class_names": valid_class_names,
        "source_mode": "folder",
        "val_split_name": "custom_split",
    }
    return categories_dir, train_dataset, val_dataset, num_classes, metadata
