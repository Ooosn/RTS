#!/usr/bin/env python3
"""Create the paper capacity/factor analysis figure from audited CSV values."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    capacity = read_csv(args.capacity)
    factors = read_csv(args.factors)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, (capacity_ax, factor_ax) = plt.subplots(
        1, 2, figsize=(7.05, 2.35), gridspec_kw={"width_ratios": [1.08, 0.92]}
    )

    fixed = [row for row in capacity if row["kind"] in {"baseline", "fixed", "fixed_default"}]
    fixed_x = [float(row["texels_m"]) for row in fixed]
    fixed_y = [float(row["psnr"]) for row in fixed]
    capacity_ax.plot(fixed_x, fixed_y, color="#2B6CB0", linewidth=1.5, zorder=2)
    for row in capacity:
        x = float(row["texels_m"])
        y = float(row["psnr"])
        kind = row["kind"]
        if kind == "adaptive":
            marker, color, size = "D", "#C53030", 38
        elif kind == "fixed_default":
            marker, color, size = "*", "#D69E2E", 88
        elif kind == "baseline":
            marker, color, size = "s", "#4A5568", 30
        else:
            marker, color, size = "o", "#2B6CB0", 32
        capacity_ax.scatter(x, y, marker=marker, color=color, s=size, zorder=3, edgecolor="white", linewidth=0.5)
        label = row["representation"].replace("Gaussian-level only", "Gaussian-level")
        offsets = {
            "Gaussian-level only": (5, 5),
            "Fixed 2x2": (4, 5),
            "Fixed 3x3": (4, 5),
            "Fixed 4x4": (-5, 9),
            "Fixed 5x5": (7, -1),
            "Forward adaptive": (5, -14),
        }
        offset = offsets[row["representation"]]
        horizontal_alignment = "right" if row["representation"] == "Fixed 4x4" else "left"
        capacity_ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            ha=horizontal_alignment,
        )

    capacity_ax.set_title("(a) Local transport capacity", loc="left", fontweight="bold")
    capacity_ax.set_xlabel("Local texels (million)")
    capacity_ax.set_ylabel("PSNR (dB)")
    capacity_ax.set_xlim(-0.16, 5.75)
    capacity_ax.set_ylim(33.50, 34.38)
    capacity_ax.yaxis.set_major_locator(MultipleLocator(0.2))
    capacity_ax.grid(axis="both", color="#D9DEE7", linewidth=0.55, alpha=0.85)
    capacity_ax.set_axisbelow(True)

    variants = [row["variant"] for row in factors]
    deltas = [float(row["delta_psnr"]) for row in factors]
    positions = list(range(len(variants)))
    colors = ["#4C78A8", "#59A14F", "#E15759", "#7A5195"]
    bars = factor_ax.barh(positions, deltas, color=colors, height=0.62)
    factor_ax.axvline(0.0, color="#2D3748", linewidth=0.8)
    factor_ax.set_yticks(positions, variants)
    factor_ax.invert_yaxis()
    factor_ax.set_xlim(-4.9, 0.0)
    factor_ax.xaxis.set_major_locator(MultipleLocator(1.0))
    factor_ax.set_xlabel(r"PSNR change (dB)")
    factor_ax.set_title("(b) Factor interventions", loc="left", fontweight="bold")
    factor_ax.grid(axis="x", color="#D9DEE7", linewidth=0.55, alpha=0.85)
    factor_ax.set_axisbelow(True)
    for bar, value in zip(bars, deltas):
        factor_ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha="left", fontsize=7.5, color="white", fontweight="bold")

    for axis in (capacity_ax, factor_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout(w_pad=1.5)
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(figure)
    print(args.out.with_suffix(".pdf"))
    print(args.out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
