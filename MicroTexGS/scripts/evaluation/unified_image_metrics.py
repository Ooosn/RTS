#!/usr/bin/env python3
"""Evaluate rendered RGB images with one shared per-frame metric implementation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_images(path: Path) -> list[Path]:
    return sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ) if path.exists() else []


def load_rgb(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = data.view(height, width, 3).permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor, (width, height)


def match_images(render_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    renders = list_images(render_dir)
    ground_truth = list_images(gt_dir)
    if not renders:
        raise ValueError(f"no rendered images in {render_dir}")
    if not ground_truth:
        raise ValueError(f"no ground-truth images in {gt_dir}")
    gt_by_name = {path.name: path for path in ground_truth}
    missing = [path.name for path in renders if path.name not in gt_by_name]
    extra = sorted(set(gt_by_name) - {path.name for path in renders})
    if missing or extra:
        raise ValueError(
            f"frame-name mismatch: missing_gt={missing[:5]} extra_gt={extra[:5]} "
            f"renders={len(renders)} gt={len(ground_truth)}"
        )
    return [(path, gt_by_name[path.name]) for path in renders]


def gaussian_window(device: torch.device, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] @ kernel_1d[None, :]).view(1, 1, size, size)
    return kernel_2d.repeat(3, 1, 1, 1)


def ssim(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=3)
    mu_y = F.conv2d(y, window, padding=padding, groups=3)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x.square(), window, padding=padding, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y.square(), window, padding=padding, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=3) - mu_xy
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.flatten(1).mean(dim=1)


def evaluate(
    render_dir: Path,
    gt_dir: Path,
    device: torch.device,
    lpips_model: torch.nn.Module,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    pairs = match_images(render_dir, gt_dir)
    window = gaussian_window(device)
    per_frame: list[dict] = []

    for offset in range(0, len(pairs), batch_size):
        batch = pairs[offset : offset + batch_size]
        render_batch = []
        gt_batch = []
        sizes = []
        for render_path, gt_path in batch:
            render, render_size = load_rgb(render_path)
            gt, gt_size = load_rgb(gt_path)
            if render_size != gt_size:
                raise ValueError(
                    f"resolution mismatch for {render_path.name}: render={render_size}, gt={gt_size}"
                )
            if sizes and render_size != sizes[0]:
                raise ValueError(f"mixed resolutions in one batch: {sizes[0]} and {render_size}")
            render_batch.append(render.clamp_(0.0, 1.0))
            gt_batch.append(gt.clamp_(0.0, 1.0))
            sizes.append(render_size)

        rendered = torch.stack(render_batch).to(device, non_blocking=True)
        target = torch.stack(gt_batch).to(device, non_blocking=True)
        with torch.no_grad():
            mse = (rendered - target).square().mean(dim=(1, 2, 3))
            psnr_values = (-10.0 * torch.log10(mse)).cpu().tolist()
            ssim_values = ssim(rendered, target, window).cpu().tolist()
            lpips_values = (
                lpips_model(rendered * 2.0 - 1.0, target * 2.0 - 1.0)
                .flatten()
                .cpu()
                .tolist()
            )

        for (render_path, _), size, psnr_value, ssim_value, lpips_value in zip(
            batch, sizes, psnr_values, ssim_values, lpips_values
        ):
            per_frame.append(
                {
                    "frame": render_path.name,
                    "width": size[0],
                    "height": size[1],
                    "psnr": float(psnr_value),
                    "ssim": float(ssim_value),
                    "lpips": float(lpips_value),
                }
            )

    count = len(per_frame)
    summary = {
        "count": count,
        "width": per_frame[0]["width"],
        "height": per_frame[0]["height"],
        "psnr": sum(row["psnr"] for row in per_frame) / count,
        "ssim": sum(row["ssim"] for row in per_frame) / count,
        "lpips": sum(row["lpips"] for row in per_frame) / count,
    }
    return summary, per_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    import lpips

    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    entries = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise SystemExit("The evaluation manifest is empty.")

    results = []
    for entry in entries:
        scene = entry["scene"]
        method = entry["method"]
        average, frames = evaluate(
            Path(entry["renders"]),
            Path(entry["gt"]),
            device,
            lpips_model,
            args.batch_size,
        )
        frame_dir = args.out / "per_frame" / scene.replace("/", "__")
        frame_dir.mkdir(parents=True, exist_ok=True)
        with (frame_dir / f"{method}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=frames[0].keys())
            writer.writeheader()
            writer.writerows(frames)
        result = {**entry, **average}
        results.append(result)
        print(
            f"{scene:28s} {method:18s} n={average['count']:4d} "
            f"{average['width']}x{average['height']} PSNR={average['psnr']:.4f} "
            f"SSIM={average['ssim']:.4f} LPIPS={average['lpips']:.4f}",
            flush=True,
        )

    fields = [
        "scene", "method", "count", "width", "height", "psnr", "ssim", "lpips", "renders", "gt"
    ]
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in results])

    with (args.out / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("| Scene | Method | PSNR | SSIM | LPIPS | Frames | Resolution |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in sorted(results, key=lambda item: (item["scene"], item["method"])):
            handle.write(
                f"| {row['scene']} | {row['method']} | {row['psnr']:.4f} | "
                f"{row['ssim']:.4f} | {row['lpips']:.4f} | {row['count']} | "
                f"{row['width']}x{row['height']} |\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
