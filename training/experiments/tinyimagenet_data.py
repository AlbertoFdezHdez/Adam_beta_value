from __future__ import annotations

from pathlib import Path

from beta_paths import default_dataset_dir, resolve_repo_path


def fix_tinyimagenet_val(root: Path) -> None:
    root = Path(root)
    val_dir = root / "val"
    img_dir = val_dir / "images"
    ann_file = val_dir / "val_annotations.txt"

    if not img_dir.exists():
        return
    if not ann_file.exists():
        raise FileNotFoundError(f"TinyImageNet validation annotations not found: {ann_file}")

    with ann_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            img, cls = line.split("\t")[:2]
            cls_dir = val_dir / cls
            src = img_dir / img
            dst = cls_dir / img
            cls_dir.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                src.rename(dst)

    if img_dir.exists():
        img_dir.rmdir()


def is_tinyimagenet_root(path: Path) -> bool:
    return (path / "train").exists() and (path / "val").exists()


def resolve_tinyimagenet_root(raw_path: str | None) -> Path:
    requested_root = resolve_repo_path(raw_path, default_dataset_dir("tinyimagenet"))
    candidates = [requested_root]
    if requested_root.name != "tiny-imagenet-200":
        candidates.append(requested_root / "tiny-imagenet-200")

    dataset_root = None
    for candidate in candidates:
        if is_tinyimagenet_root(candidate):
            dataset_root = candidate
            break

    if dataset_root is None:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "TinyImageNet not found. Expected the extracted dataset under one of: "
            f"{checked}"
        )

    fix_tinyimagenet_val(dataset_root)
    return dataset_root
