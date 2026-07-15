#!/usr/bin/env python3
"""Collect structured fixed-versus-RTD render benchmark records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PREFIX = "BENCHMARK_JSON "


def load_benchmark(path: Path) -> dict:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(PREFIX):
            records.append(json.loads(line[len(PREFIX) :]))
    if len(records) != 1:
        raise ValueError(f"Expected exactly one benchmark record in {path}, found {len(records)}")
    return records[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise SystemExit("The benchmark manifest is empty.")

    rows = []
    for entry in entries:
        model = Path(entry["model"])
        record = load_benchmark(Path(entry["log"]))
        rows.append({
            "scene": entry["scene"],
            "method": entry["method"],
            "views": int(record["views"]),
            "timed_frames": int(record["timed_frames"]),
            "average_frame_ms": float(record["average_frame_ms"]),
            "fps": float(record["fps"]),
            "peak_allocated_mb": float(record["peak_allocated_mb"]),
            "peak_reserved_mb": float(record["peak_reserved_mb"]),
            "checkpoint_mb": (model / "chkpnt100000.pth").stat().st_size / (1024.0 * 1024.0),
            "model": str(model),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = args.out.with_suffix(".md")
    with markdown.open("w", encoding="utf-8") as handle:
        handle.write("| Scene | Method | ms/frame | FPS | Peak allocated | Checkpoint |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['scene']} | {row['method']} | {row['average_frame_ms']:.3f} | "
                f"{row['fps']:.2f} | {row['peak_allocated_mb']:.1f} MB | "
                f"{row['checkpoint_mb']:.1f} MB |\n"
            )
    print(args.out)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
