from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "beta_sweep_curves.tsv"
OUT_DIR = ROOT / "outputs" / "figures"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


@dataclass(frozen=True)
class CurveRow:
    split: str
    label: str
    model: str
    dataset: str
    beta: float
    mean_loss: float
    lower_std: float
    upper_std: float
    seed_losses: tuple[float, ...]


TRAIN_ORDER = [
    "ResNet50 on Food-101",
    "ResNet50 on ImageNet100",
    "ViT-B/16 on CIFAR-100",
    "ViT-B/16 on TinyImageNet",
    "NanoGPT on WikiText-103",
    "NanoGPT on OpenWebText",
    "Llama60M on C4",
    "Llama60M on SlimPajama-6B",
]

HELDOUT_ORDER = [
    "T5-small on BookCorpus",
    "Swin-T on Caltech-256",
    "EfficientNet-B0 on Stanford Cars",
]


def read_rows() -> dict[str, list[CurveRow]]:
    out: dict[str, list[CurveRow]] = {}
    with DATA_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            seed_losses = tuple(float(x) for x in row["seed_best_val_losses"].split(";") if x)
            item = CurveRow(
                split=row["split"],
                label=row["label"],
                model=row["model"],
                dataset=row["dataset"],
                beta=float(row["beta"]),
                mean_loss=float(row["mean_best_val_loss"]),
                lower_std=float(row["lower_std"]),
                upper_std=float(row["upper_std"]),
                seed_losses=seed_losses,
            )
            out.setdefault(item.label, []).append(item)
    for rows in out.values():
        rows.sort(key=lambda x: x.beta)
    return out


def x_from_beta(beta: float) -> float:
    return -np.log10(1.0 - beta)


def latex_title(label: str) -> str:
    model, dataset = label.split(" on ", 1)

    def clean(text: str) -> str:
        return text.replace("-", r"{-}").replace(" ", r"\ ")

    return rf"$\mathrm{{{clean(model)}}}\ \mathrm{{on}}\ \mathrm{{{clean(dataset)}}}$"


def beta_tick_label(beta: float) -> str:
    return rf"${beta:.3f}$"


def plot_curve(ax, rows: list[CurveRow], show_ylabel: bool) -> None:
    betas = np.asarray([r.beta for r in rows], dtype=float)
    xs = np.asarray([x_from_beta(b) for b in betas], dtype=float)
    means = np.asarray([r.mean_loss for r in rows], dtype=float)
    lows = np.asarray([r.lower_std for r in rows], dtype=float)
    highs = np.asarray([r.upper_std for r in rows], dtype=float)
    best_idx = int(np.argmin(means))

    ax.plot(xs, means, color="#0F766E", linewidth=2.5, marker="o", markersize=4.8, zorder=4)
    ax.fill_between(xs, means - lows, means + highs, color="#0F766E", alpha=0.16, linewidth=0, zorder=2)

    for x, row in zip(xs, rows):
        if len(row.seed_losses) <= 1:
            continue
        jitters = np.linspace(-0.012, 0.012, len(row.seed_losses))
        ax.scatter(
            np.full(len(row.seed_losses), x) + jitters,
            row.seed_losses,
            s=16,
            color="#0F766E",
            alpha=0.30,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    ax.scatter([xs[best_idx]], [means[best_idx]], color="#B91C1C", s=56, marker="*", zorder=6)
    ax.axvline(xs[best_idx], color="#B91C1C", linewidth=0.9, alpha=0.25, zorder=1)
    ax.set_title(latex_title(rows[0].label), fontsize=12.7, pad=6)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.set_xlim(xs.min() - 0.03, xs.max() + 0.03)
    ax.set_xticks(xs)
    ax.set_xticklabels([beta_tick_label(beta) for beta in betas], rotation=35, ha="right", fontsize=8.8)
    ax.set_ylabel(r"$\mathrm{Best\ validation\ loss}$" if show_ylabel else "", fontsize=11)


def make_train(rows_by_label: dict[str, list[CurveRow]]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10.1, 11.2))
    for idx, (ax, label) in enumerate(zip(axes.flat, TRAIN_ORDER)):
        plot_curve(ax, rows_by_label[label], show_ylabel=(idx % 2 == 0))
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\beta\ \mathrm{value}$", fontsize=11)
    fig.tight_layout(rect=(0.035, 0.035, 0.995, 0.995))
    fig.savefig(OUT_DIR / "beta_sweep_curves_train.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "beta_sweep_curves_train.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_heldout(rows_by_label: dict[str, list[CurveRow]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.55))
    for idx, (ax, label) in enumerate(zip(axes, HELDOUT_ORDER)):
        plot_curve(ax, rows_by_label[label], show_ylabel=(idx == 0))
        ax.set_xlabel(r"$\beta\ \mathrm{value}$", fontsize=11)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.24, top=0.87, wspace=0.22)
    fig.savefig(OUT_DIR / "beta_sweep_curves_validation.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "beta_sweep_curves_validation.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    make_train(rows)
    make_heldout(rows)
    print(OUT_DIR / "beta_sweep_curves_train.pdf")
    print(OUT_DIR / "beta_sweep_curves_validation.pdf")


if __name__ == "__main__":
    main()
