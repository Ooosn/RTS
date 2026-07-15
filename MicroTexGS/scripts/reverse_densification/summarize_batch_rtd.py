#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_baseline(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["scene"]: row
            for row in rows
            if row.get("method", "").strip().lower() == "microtexgs"
        }


def load_unified_metrics(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row["scene"], row["method"]): row
            for row in csv.DictReader(handle)
        }


def scene_from_run(run_dir: Path) -> str:
    parts = run_dir.parts
    if "microtexgs" not in parts:
        return run_dir.name
    index = len(parts) - 1 - list(reversed(parts)).index("microtexgs")
    return "/".join(parts[index + 1 :])


def find_runs(run_roots: list[Path]) -> list[Path]:
    found = []
    for root in run_roots:
        if (root / "convergence" / "eval_metrics.jsonl").exists():
            found.append(root)
            continue
        for metrics in root.rglob("convergence/eval_metrics.jsonl"):
            found.append(metrics.parent.parent)
    return sorted(set(found), key=lambda path: scene_from_run(path))


def final_row(rows: list[dict], split: str) -> dict:
    selected = [row for row in rows if row.get("split") == split]
    return max(selected, key=lambda row: int(row.get("iteration", -1)), default={})


def final_arch(rows: list[dict]) -> dict:
    checkpoints = [row for row in rows if row.get("event") == "checkpoint"]
    if checkpoints:
        return max(checkpoints, key=lambda row: int(row.get("iteration", -1)))
    return rows[-1] if rows else {}


def first_arch(rows: list[dict]) -> dict:
    starts = [row for row in rows if row.get("event") == "train_start"]
    return starts[0] if starts else (rows[0] if rows else {})


def compact_events(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("event") == "rtd_compress"]


def compact_summary(events: list[dict]) -> str:
    values = []
    for row in events:
        log = row.get("last_rtd_compress_log", {})
        values.append(
            f"{int(row.get('iteration', 0))}:"
            f"{int(log.get('compressed', 0))}:"
            f"{int(log.get('before_texels', 0))}->{int(log.get('after_texels', 0))}"
        )
    return ";".join(values)


def summarize(
    run_dir: Path,
    baseline: dict[str, dict],
    unified_metrics: dict[tuple[str, str], dict],
) -> dict:
    convergence = run_dir / "convergence"
    eval_rows = read_jsonl(convergence / "eval_metrics.jsonl")
    arch_rows = read_jsonl(convergence / "texture_architecture.jsonl")
    test = final_row(eval_rows, "test")
    train = final_row(eval_rows, "train")
    start = first_arch(arch_rows)
    final = final_arch(arch_rows)
    events = compact_events(arch_rows)
    scene = scene_from_run(run_dir)
    baseline_row = baseline.get(scene, {})
    fixed_eval = unified_metrics.get((scene, "MicroTexGS-fixed"), {})
    rtd_eval = unified_metrics.get((scene, "MicroTexGS-RTD"), {})

    start_texels = int(start.get("texels", 0) or 0)
    final_texels = int(final.get("texels", 0) or 0)
    savings = 0.0 if start_texels <= 0 else 1.0 - final_texels / start_texels
    checkpoint = run_dir / "chkpnt100000.pth"
    checkpoint_mb = checkpoint.stat().st_size / (1024.0 * 1024.0) if checkpoint.exists() else 0.0
    baseline_psnr = float(fixed_eval.get("psnr", baseline_row.get("psnr", 0.0)) or 0.0)
    baseline_ssim = float(fixed_eval.get("ssim", baseline_row.get("ssim", 0.0)) or 0.0)
    baseline_lpips = float(fixed_eval.get("lpips", baseline_row.get("lpips", 0.0)) or 0.0)
    psnr = float(rtd_eval.get("psnr", test.get("psnr", 0.0)) or 0.0)
    ssim = float(rtd_eval.get("ssim", test.get("ssim", 0.0)) or 0.0)
    lpips = float(rtd_eval.get("lpips", test.get("lpips", 0.0)) or 0.0)

    return {
        "scene": scene,
        "iteration": int(test.get("iteration", 0) or 0),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "train_psnr": float(train.get("psnr", 0.0) or 0.0),
        "baseline_psnr": baseline_psnr,
        "delta_psnr": psnr - baseline_psnr,
        "baseline_ssim": baseline_ssim,
        "delta_ssim": ssim - baseline_ssim,
        "baseline_lpips": baseline_lpips,
        "delta_lpips": lpips - baseline_lpips,
        "eval_count": int(rtd_eval.get("count", 0) or 0),
        "eval_width": int(rtd_eval.get("width", 0) or 0),
        "eval_height": int(rtd_eval.get("height", 0) or 0),
        "gaussians": int(final.get("gaussians", 0) or 0),
        "start_texels": start_texels,
        "final_texels": final_texels,
        "texel_reduction": start_texels - final_texels,
        "texel_reduction_pct": 100.0 * savings,
        "avg_texels_per_gaussian": float(final.get("avg_texels_per_gaussian", 0.0) or 0.0),
        "resolution_hist": json.dumps(final.get("resolution_hist", []), ensure_ascii=False),
        "checkpoint_mb": checkpoint_mb,
        "compact_events": compact_summary(events),
        "run": str(run_dir),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| Scene | Fixed PSNR | RTD PSNR | Delta | Texels | Reduction | Checkpoint |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['scene']} | {row['baseline_psnr']:.3f} | {row['psnr']:.3f} | "
                f"{row['delta_psnr']:+.3f} | {row['final_texels']:,} | "
                f"{row['texel_reduction_pct']:.1f}% | {row['checkpoint_mb']:.1f} MB |\n"
            )
        if rows:
            handle.write(
                f"| **Average** | {mean(row['baseline_psnr'] for row in rows):.3f} | "
                f"{mean(row['psnr'] for row in rows):.3f} | "
                f"{mean(row['delta_psnr'] for row in rows):+.3f} | -- | "
                f"{mean(row['texel_reduction_pct'] for row in rows):.1f}% | -- |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--unified-metrics-csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_baseline(args.baseline_csv)
    unified_metrics = load_unified_metrics(args.unified_metrics_csv)
    runs = find_runs(args.run_roots)
    rows = [summarize(run, baseline, unified_metrics) for run in runs]
    if not rows:
        raise SystemExit("No completed RTD runs with convergence metrics found.")
    write_csv(args.out, rows)
    write_markdown(args.out.with_suffix(".md"), rows)
    print(args.out)
    print(args.out.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
