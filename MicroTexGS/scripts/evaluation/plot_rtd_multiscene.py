#!/usr/bin/env python3
"""Plot the multi-scene reverse-densification result from an audited summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def short_scene(scene: str) -> str:
    return scene.split("/", 1)[-1]


def resolution_percentages(row: dict[str, str]) -> list[float]:
    histogram = {int(resolution): int(count) for resolution, count in json.loads(row["resolution_hist"])}
    total = sum(histogram.values())
    if total <= 0:
        raise ValueError(f"Empty resolution histogram for {row['scene']}")
    return [100.0 * histogram.get(resolution, 0) / total for resolution in (1, 2, 3, 4)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", choices=("single", "wide"), default="single")
    args = parser.parse_args()

    rows = read_rows(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if args.layout == "single":
        figure, (tradeoff_ax, layout_ax) = plt.subplots(
            2,
            1,
            figsize=(3.35, 4.35),
            gridspec_kw={"height_ratios": [0.92, 1.08]},
        )
    else:
        figure, (tradeoff_ax, layout_ax) = plt.subplots(
            1,
            2,
            figsize=(7.05, 2.55),
            gridspec_kw={"width_ratios": [0.96, 1.04]},
        )

    reductions = [float(row["texel_reduction_pct"]) for row in rows]
    quality = [float(row["delta_psnr"]) for row in rows]
    labels = [short_scene(row["scene"]) for row in rows]
    colors = ["#2B6CB0", "#2F855A", "#B7791F", "#805AD5", "#C53030", "#319795"]

    tradeoff_ax.scatter(reductions, quality, s=34, color=colors[: len(rows)], edgecolor="white", linewidth=0.55, zorder=3)
    label_offsets = {
        "Fish": (4, 8),
        "CupFabric": (5, -15),
        "Container": (-48, 6),
        "AnisoMetal": (4, 4),
        "FurBall": (4, 4),
        "Boot": (4, 4),
    }
    for x_value, y_value, label in zip(reductions, quality, labels):
        tradeoff_ax.annotate(
            label,
            (x_value, y_value),
            xytext=label_offsets.get(label, (4, 4)),
            textcoords="offset points",
            fontsize=6.8,
        )
    tradeoff_ax.axhline(0.0, color="#4A5568", linewidth=0.8, linestyle="--")
    tradeoff_ax.set_title("(a) Quality--capacity trade-off", loc="left", fontweight="bold")
    tradeoff_ax.set_xlabel("Local-texel reduction")
    tradeoff_ax.set_ylabel(r"$\Delta$PSNR from fixed $4\!\times\!4$ (dB)")
    tradeoff_ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    x_margin = max(2.0, 0.08 * (max(reductions) - min(reductions) or 1.0))
    y_margin = max(0.03, 0.15 * (max(quality) - min(quality) or 0.2))
    tradeoff_ax.set_xlim(max(0.0, min(reductions) - x_margin), max(reductions) + x_margin)
    tradeoff_ax.set_ylim(min(min(quality) - y_margin, -0.05), max(max(quality) + y_margin, 0.05))
    tradeoff_ax.grid(color="#D9DEE7", linewidth=0.55, alpha=0.85)
    tradeoff_ax.set_axisbelow(True)

    positions = list(range(len(rows)))
    bottoms = [0.0] * len(rows)
    resolution_colors = ["#DCEAF7", "#8DB9DD", "#4C83B6", "#1F4E79"]
    percentages = [resolution_percentages(row) for row in rows]
    for resolution_index, resolution in enumerate((1, 2, 3, 4)):
        values = [scene_values[resolution_index] for scene_values in percentages]
        layout_ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.72,
            color=resolution_colors[resolution_index],
            label=rf"${resolution}\!\times\!{resolution}$",
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    layout_ax.set_title("(b) Final local-grid allocation", loc="left", fontweight="bold")
    layout_ax.set_ylabel("Fraction of Gaussians")
    layout_ax.set_xticks(positions, labels, rotation=28, ha="right")
    layout_ax.set_ylim(0.0, 100.0)
    layout_ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    layout_ax.grid(axis="y", color="#D9DEE7", linewidth=0.55, alpha=0.85)
    layout_ax.set_axisbelow(True)
    layout_ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=1.1, columnspacing=0.8)

    for axis in (tradeoff_ax, layout_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    if args.layout == "single":
        figure.tight_layout(h_pad=1.25)
    else:
        figure.tight_layout(w_pad=1.4)
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(figure)
    print(args.out.with_suffix(".pdf"))
    print(args.out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
