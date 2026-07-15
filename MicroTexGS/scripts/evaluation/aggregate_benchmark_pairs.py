#!/usr/bin/env python3
"""Aggregate audited fixed-versus-RTD render benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


METHOD_FIXED = "MicroTexGS-fixed"
METHOD_RTD = "MicroTexGS-RTD"


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("No benchmark rows were provided.")
    return rows


def pair_rows(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    by_scene: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        scene = row["scene"]
        method = row["method"]
        if method not in (METHOD_FIXED, METHOD_RTD):
            raise ValueError(f"Unexpected method for {scene}: {method}")
        if method in by_scene.setdefault(scene, {}):
            raise ValueError(f"Duplicate benchmark row for {scene}, {method}")
        by_scene[scene][method] = row

    paired = []
    for scene in sorted(by_scene):
        methods = by_scene[scene]
        if set(methods) != {METHOD_FIXED, METHOD_RTD}:
            raise ValueError(f"Incomplete fixed/RTD pair for {scene}: {sorted(methods)}")
        fixed = methods[METHOD_FIXED]
        rtd = methods[METHOD_RTD]
        if fixed["views"] != rtd["views"] or fixed["timed_frames"] != rtd["timed_frames"]:
            raise ValueError(f"Mismatched benchmark budgets for {scene}")

        fixed_fps = float(fixed["fps"])
        rtd_fps = float(rtd["fps"])
        fixed_peak = float(fixed["peak_allocated_mb"])
        rtd_peak = float(rtd["peak_allocated_mb"])
        fixed_checkpoint = float(fixed["checkpoint_mb"])
        rtd_checkpoint = float(rtd["checkpoint_mb"])
        paired.append({
            "scene": scene,
            "views": int(fixed["views"]),
            "timed_frames": int(fixed["timed_frames"]),
            "fixed_fps": fixed_fps,
            "rtd_fps": rtd_fps,
            "fps_change_pct": 100.0 * (rtd_fps / fixed_fps - 1.0),
            "fixed_peak_allocated_mb": fixed_peak,
            "rtd_peak_allocated_mb": rtd_peak,
            "peak_reduction_pct": 100.0 * (1.0 - rtd_peak / fixed_peak),
            "fixed_checkpoint_mb": fixed_checkpoint,
            "rtd_checkpoint_mb": rtd_checkpoint,
            "checkpoint_reduction_pct": 100.0 * (1.0 - rtd_checkpoint / fixed_checkpoint),
        })
    return paired


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| Scene | Fixed FPS | RTD FPS | FPS change | Peak reduction | Checkpoint reduction |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['scene']} | {row['fixed_fps']:.2f} | {row['rtd_fps']:.2f} | "
                f"{row['fps_change_pct']:+.1f}% | {row['peak_reduction_pct']:.1f}% | "
                f"{row['checkpoint_reduction_pct']:.1f}% |\n"
            )
        handle.write(
            f"| **Average** | {mean(float(row['fixed_fps']) for row in rows):.2f} | "
            f"{mean(float(row['rtd_fps']) for row in rows):.2f} | "
            f"{mean(float(row['fps_change_pct']) for row in rows):+.1f}% | "
            f"{mean(float(row['peak_reduction_pct']) for row in rows):.1f}% | "
            f"{mean(float(row['checkpoint_reduction_pct']) for row in rows):.1f}% |\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = pair_rows(read_rows(args.inputs))
    write_csv(args.out, rows)
    write_markdown(args.out.with_suffix(".md"), rows)
    print(args.out)
    print(args.out.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
