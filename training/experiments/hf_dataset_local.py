from __future__ import annotations

from pathlib import Path

from datasets import DownloadConfig, load_dataset, load_from_disk


def _is_saved_dataset(path: Path) -> bool:
    return (path / "dataset_dict.json").exists() or (path / "state.json").exists()


def load_local_dataset_dict(
    dataset_dir: str | Path,
    dataset_name: str,
    config_name: str | None = None,
):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Local dataset directory not found: {dataset_dir}")

    if _is_saved_dataset(dataset_dir):
        dataset = load_from_disk(str(dataset_dir))
        source_mode = "disk"
    else:
        kwargs = {
            "cache_dir": str(dataset_dir),
            "download_config": DownloadConfig(local_files_only=True),
        }
        if config_name is None:
            dataset = load_dataset(dataset_name, **kwargs)
        else:
            dataset = load_dataset(dataset_name, config_name, **kwargs)
        source_mode = "cache"

    if not hasattr(dataset, "keys"):
        raise ValueError(f"Expected a DatasetDict-like object at {dataset_dir}, got {type(dataset).__name__}")

    return dataset, source_mode
