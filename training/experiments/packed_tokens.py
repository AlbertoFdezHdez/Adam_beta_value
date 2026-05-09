from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def safe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return None


def packed_paths(root_dir: str | Path, tag: str) -> tuple[Path, Path]:
    root_dir = Path(root_dir)
    return root_dir / f"{tag}.bin", root_dir / f"{tag}.len.npy"


def iter_text_batches(dataset_split, text_key: str, batch_docs: int):
    total_docs = safe_len(dataset_split)
    if total_docs is not None:
        for start in range(0, total_docs, batch_docs):
            stop = min(total_docs, start + batch_docs)
            batch = dataset_split[start:stop]
            yield [text if isinstance(text, str) else "" for text in batch[text_key]]
        return

    if hasattr(dataset_split, "iter"):
        for batch in dataset_split.iter(batch_size=batch_docs):
            yield [text if isinstance(text, str) else "" for text in batch[text_key]]
        return

    batch = []
    for row in dataset_split:
        batch.append(row.get(text_key, ""))
        if len(batch) >= batch_docs:
            yield [text if isinstance(text, str) else "" for text in batch]
            batch = []
    if batch:
        yield [text if isinstance(text, str) else "" for text in batch]


def write_packed_split(
    dataset_split,
    tokenizer,
    output_dir: str | Path,
    tag: str,
    *,
    text_key: str = "text",
    batch_docs: int = 1024,
    overwrite: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, int | float | str]:
    log = log or (lambda message: None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path, len_path = packed_paths(output_dir, tag)

    if bin_path.exists() and len_path.exists() and not overwrite:
        total_tokens = int(np.load(len_path))
        log(
            f"Packed split already exists | tag={tag} | docs={safe_len(dataset_split) if safe_len(dataset_split) is not None else 'unknown'} "
            f"| tokens={total_tokens} | bin={bin_path} | len={len_path}"
        )
        return {
            "tag": tag,
            "bin_path": str(bin_path),
            "len_path": str(len_path),
            "total_tokens": total_tokens,
            "docs": safe_len(dataset_split) if safe_len(dataset_split) is not None else -1,
            "elapsed_sec": 0.0,
            "skipped": True,
        }

    if getattr(tokenizer, "vocab_size", 0) > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size} exceeds uint16 capacity; "
            "the packed format must be widened before use."
        )

    tmp_bin = bin_path.with_suffix(bin_path.suffix + ".tmp")
    tmp_len = len_path.with_suffix(len_path.suffix + ".tmp")
    if tmp_bin.exists():
        tmp_bin.unlink()
    if tmp_len.exists():
        tmp_len.unlink()

    total_docs = safe_len(dataset_split)
    total_tokens = 0
    seen_docs = 0
    nonempty_docs = 0
    t0 = time.time()
    next_log = t0 + 60.0

    log(
        f"Packing split start | tag={tag} | docs={total_docs if total_docs is not None else 'unknown'} "
        f"| batch_docs={batch_docs} | output_dir={output_dir}"
    )

    with tmp_bin.open("wb") as handle:
        for texts in iter_text_batches(dataset_split, text_key=text_key, batch_docs=batch_docs):
            encodings = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            for ids in encodings["input_ids"]:
                if not ids:
                    continue
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(handle)
                total_tokens += int(arr.size)
                nonempty_docs += 1

            seen_docs += len(texts)
            now = time.time()
            if now >= next_log:
                rate = seen_docs / max(1e-9, now - t0)
                log(
                    f"Packing split progress | tag={tag} | docs={seen_docs}"
                    + (f"/{total_docs}" if total_docs is not None else "")
                    + f" | nonempty_docs={nonempty_docs} | tokens={total_tokens} | docs_per_sec={rate:.2f}"
                )
                next_log = now + 60.0

    with tmp_len.open("wb") as handle:
        np.save(handle, np.array(total_tokens, dtype=np.int64))
    os.replace(tmp_bin, bin_path)
    os.replace(tmp_len, len_path)
    elapsed_sec = time.time() - t0
    log(
        f"Packing split done | tag={tag} | docs={seen_docs} | nonempty_docs={nonempty_docs} "
        f"| tokens={total_tokens} | elapsed_sec={elapsed_sec:.1f} | bin={bin_path} | len={len_path}"
    )
    return {
        "tag": tag,
        "bin_path": str(bin_path),
        "len_path": str(len_path),
        "total_tokens": total_tokens,
        "docs": seen_docs,
        "elapsed_sec": elapsed_sec,
        "skipped": False,
    }


class PackedTokens:
    def __init__(self, root_dir: str | Path, tag: str, block_size: int, log: Callable[[str], None] | None = None):
        log = log or (lambda message: None)
        self.block_size = int(block_size)
        self.bin_path, self.len_path = packed_paths(root_dir, tag)
        if not self.bin_path.exists() or not self.len_path.exists():
            raise FileNotFoundError(
                f"Packed tokens not found for tag={tag}. Missing files: {self.bin_path}, {self.len_path}. "
                "Run the corresponding preprocessing script first to generate the packed artifacts."
            )

        self.length = int(np.load(self.len_path))
        self.mem = np.memmap(self.bin_path, mode="r", dtype=np.uint16, shape=(self.length,))
        log(
            f"Packed tokens ready | tag={tag} | tokens={self.length} | "
            f"bin={self.bin_path} | len={self.len_path}"
        )

    def sample_batch(self, batch_size: int, device, generator=None):
        max_start = self.length - self.block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"Packed stream too short for block_size={self.block_size}: total_tokens={self.length}"
            )

        ix = torch.randint(0, max_start, (batch_size,), device=device, generator=generator)
        x = torch.empty((batch_size, self.block_size), dtype=torch.long, device=device)
        for i, start in enumerate(ix.tolist()):
            x[i] = torch.from_numpy(self.mem[start : start + self.block_size].astype(np.int64)).to(device)
        return x
