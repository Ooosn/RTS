import math
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SUBMODULE_PATHS = [
    os.path.join(_REPO_ROOT, "submodules", "diff-surfel-rasterization-shadow"),
]
for _path in reversed(_SUBMODULE_PATHS):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from diff_surfel_rasterization_shadow import GaussianRasterizationSettings as ShadowRasterSettings
from diff_surfel_rasterization_shadow import GaussianRasterizer as ShadowRasterizer
from gaussian_renderer.textured import TextureRenderInputs, rasterize_with_texture_module
from utils.general_utils import build_rotation
from utils.graphics_utils import getProjectionMatrix


def _dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def _safe_normalize(x):
    return F.normalize(x, dim=-1, eps=1e-8)


def _NdotWi(nrm, wi, elu, a):
    tmp = a * (1.0 - 1.0 / math.e)
    return (elu(_dot(nrm, wi)) + tmp) / (1.0 + tmp)


def _project_to_local(vec, local_axises):
    return torch.einsum("ki,kij->kj", vec, local_axises)


def _canonicalize_quaternion(q):
    q = q.to(dtype=torch.float32)
    finite = torch.isfinite(q).all(dim=1, keepdim=True)
    norm = q.norm(dim=1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[:, 0] = 1.0
    q = torch.where(finite & (norm > 1e-8), q, identity)
    q = torch.where(q[:, 0:1] < 0.0, -q, q)
    return q


def _frame_from_texture_local_q(fallback_axises, texture_local_q):
    if texture_local_q is None or texture_local_q.numel() == 0:
        return fallback_axises
    return build_rotation(_canonicalize_quaternion(texture_local_q)).to(
        device=fallback_axises.device,
        dtype=fallback_axises.dtype,
    )


def _require_point_shadow(shadow_pkg):
    point_shadow = None if shadow_pkg is None else shadow_pkg.get("per_point_shadow")
    if point_shadow is None:
        raise RuntimeError("Texture MBRDF requires per_point_shadow from the texture-aware shadow pass.")
    return point_shadow


def _mean_dynamic_texture_values(flat_values, texture_dims):
    if texture_dims.numel() == 0:
        return flat_values
    counts = (texture_dims[:, 0].to(torch.long) * texture_dims[:, 1].to(torch.long)).clamp_min(1)
    offsets = texture_dims[:, 2].to(torch.long)
    values = flat_values.reshape(flat_values.shape[0], -1)
    out = torch.empty((texture_dims.shape[0], values.shape[1]), dtype=values.dtype, device=values.device)
    point_ids = torch.arange(texture_dims.shape[0], dtype=torch.long, device=values.device)
    texel_point_ids = torch.repeat_interleave(point_ids, counts)
    total_texels = int(counts.sum().item())
    if total_texels > 0:
        starts = torch.cumsum(counts, dim=0) - counts
        local_offsets = torch.arange(total_texels, dtype=torch.long, device=values.device) - torch.repeat_interleave(starts, counts)
        flat_ids = torch.repeat_interleave(offsets, counts) + local_offsets
        accum = torch.zeros_like(out)
        accum.index_add_(0, texel_point_ids, values[flat_ids])
        out = accum / counts.to(values.dtype).unsqueeze(-1)
    return out.reshape(texture_dims.shape[0], *flat_values.shape[1:])


def _sum_dynamic_texture_values(flat_values, texture_dims):
    if texture_dims.numel() == 0:
        return flat_values
    counts = (texture_dims[:, 0].to(torch.long) * texture_dims[:, 1].to(torch.long)).clamp_min(1)
    offsets = texture_dims[:, 2].to(torch.long)
    values = flat_values.reshape(flat_values.shape[0], -1)
    out = torch.zeros((texture_dims.shape[0], values.shape[1]), dtype=values.dtype, device=values.device)
    point_ids = torch.arange(texture_dims.shape[0], dtype=torch.long, device=values.device)
    texel_point_ids = torch.repeat_interleave(point_ids, counts)
    total_texels = int(counts.sum().item())
    if total_texels > 0:
        starts = torch.cumsum(counts, dim=0) - counts
        local_offsets = torch.arange(total_texels, dtype=torch.long, device=values.device) - torch.repeat_interleave(starts, counts)
        flat_ids = torch.repeat_interleave(offsets, counts) + local_offsets
        out.index_add_(0, texel_point_ids, values[flat_ids])
    return out.reshape(texture_dims.shape[0], *flat_values.shape[1:])


def _sum_dynamic_texture_values_from_ids(flat_values, texture_dims, counts, texel_ids, flat_ids, contiguous_flat=False):
    if texture_dims.numel() == 0:
        return flat_values
    values = flat_values.reshape(flat_values.shape[0], -1)
    out = torch.zeros((texture_dims.shape[0], values.shape[1]), dtype=values.dtype, device=values.device)
    if contiguous_flat:
        out.index_add_(0, texel_ids, values)
    else:
        out.index_add_(0, texel_ids, values[flat_ids])
    return out.reshape(texture_dims.shape[0], *flat_values.shape[1:])


def _mean_dynamic_texture_values_from_ids(flat_values, texture_dims, counts, texel_ids, flat_ids, contiguous_flat=False):
    if texture_dims.numel() == 0:
        return flat_values
    values = flat_values.reshape(flat_values.shape[0], -1)
    out = torch.empty((texture_dims.shape[0], values.shape[1]), dtype=values.dtype, device=values.device)
    accum = torch.zeros_like(out)
    if contiguous_flat:
        accum.index_add_(0, texel_ids, values)
    else:
        accum.index_add_(0, texel_ids, values[flat_ids])
    out = accum / counts.to(values.dtype).unsqueeze(-1)
    return out.reshape(texture_dims.shape[0], *flat_values.shape[1:])


def _dynamic_texture_flat_ids_impl(texture_dims, device):
    counts = (texture_dims[:, 0].to(torch.long) * texture_dims[:, 1].to(torch.long)).clamp_min(1)
    offsets = texture_dims[:, 2].to(torch.long)
    point_ids = torch.arange(texture_dims.shape[0], dtype=torch.long, device=device)
    texel_ids = torch.repeat_interleave(point_ids, counts.to(device))
    total_texels = int(counts.sum().item())
    expected_offsets = torch.zeros_like(offsets)
    if offsets.numel() > 1:
        expected_offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    contiguous_flat = torch.equal(offsets, expected_offsets)
    if contiguous_flat:
        flat_ids = torch.arange(total_texels, dtype=torch.long, device=device)
    else:
        starts = torch.cumsum(counts, dim=0) - counts
        local_offsets = torch.arange(total_texels, dtype=torch.long, device=device) - torch.repeat_interleave(starts.to(device), counts.to(device))
        flat_ids = torch.repeat_interleave(offsets.to(device), counts.to(device)) + local_offsets
    return (counts.to(device), texel_ids, flat_ids, total_texels), contiguous_flat


def _dynamic_texture_flat_ids(texture_dims, device):
    value, _ = _dynamic_texture_flat_ids_impl(texture_dims, device)
    return value


def _cached_dynamic_texture_flat_ids_with_layout(gau, texture_dims, device, expected_total_texels=None):
    device = torch.device(device)
    version = int(getattr(texture_dims, "_version", 0))
    layout_version = int(getattr(gau, "_dynamic_texture_layout_version", 0))
    key = (
        id(texture_dims),
        int(texture_dims.data_ptr()),
        tuple(texture_dims.shape),
        version,
        layout_version,
        str(device),
    )
    cache = getattr(gau, "_dynamic_texture_flat_ids_cache", None)
    if isinstance(cache, dict) and cache.get("key") == key:
        value = cache["value"]
        cached_total = int(value[3])
        cache_shape_ok = (
            value[0].numel() == texture_dims.shape[0]
            and value[1].numel() == cached_total
            and value[2].numel() == cached_total
            and (expected_total_texels is None or cached_total == int(expected_total_texels))
        )
        if cache_shape_ok:
            return value, bool(cache.get("contiguous_flat", False))
    value, contiguous_flat = _dynamic_texture_flat_ids_impl(texture_dims, device)
    if int(value[1].numel()) != int(value[3]) or int(value[2].numel()) != int(value[3]):
        raise RuntimeError(
            "Dynamic texture layout cache construction failed: "
            f"texel_ids={int(value[1].numel())}, flat_ids={int(value[2].numel())}, "
            f"total_texels={int(value[3])}."
        )
    if expected_total_texels is not None and int(value[3]) != int(expected_total_texels):
        raise RuntimeError(
            "Dynamic texture layout mismatch: "
            f"texture_dims imply {int(value[3])} texels, "
            f"but flat texture tensor has {int(expected_total_texels)} texels."
        )
    gau._dynamic_texture_flat_ids_cache = {
        "key": key,
        "value": value,
        "contiguous_flat": bool(contiguous_flat),
        "texture_dims_ref": texture_dims,
    }
    return value, bool(contiguous_flat)


def _cached_dynamic_texture_flat_ids(gau, texture_dims, device):
    value, _ = _cached_dynamic_texture_flat_ids_with_layout(gau, texture_dims, device)
    return value


def _cached_mean_dynamic_texture_values(gau, flat_values, texture_dims):
    (counts, texel_ids, flat_ids, _), contiguous_flat = _cached_dynamic_texture_flat_ids_with_layout(
        gau,
        texture_dims,
        flat_values.device,
        expected_total_texels=flat_values.shape[0],
    )
    return _mean_dynamic_texture_values_from_ids(
        flat_values,
        texture_dims,
        counts,
        texel_ids,
        flat_ids,
        contiguous_flat=contiguous_flat,
    )


def _cached_sum_dynamic_texture_values(gau, flat_values, texture_dims):
    (counts, texel_ids, flat_ids, _), contiguous_flat = _cached_dynamic_texture_flat_ids_with_layout(
        gau,
        texture_dims,
        flat_values.device,
        expected_total_texels=flat_values.shape[0],
    )
    return _sum_dynamic_texture_values_from_ids(
        flat_values,
        texture_dims,
        counts,
        texel_ids,
        flat_ids,
        contiguous_flat=contiguous_flat,
    )


_SHADOW_FILL_DISTANCE_CACHE = {}


def _build_square_texture_dims(resolutions):
    resolutions = resolutions.to(dtype=torch.int32)
    counts = resolutions.to(torch.long) * resolutions.to(torch.long)
    offsets = torch.zeros_like(resolutions, dtype=torch.int32)
    if resolutions.numel() > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0).to(torch.int32)
    return torch.stack([resolutions, resolutions, offsets], dim=1)


def _texture_shadow_spatial_enabled(gau):
    return bool(getattr(gau, "texture_shadow_spatial_resolution", False))


def _build_spatial_shadow_dims(gau):
    texel_size = max(float(getattr(gau, "texture_shadow_spatial_texel_size", 0.01)), 1e-8)
    min_res = max(1, int(getattr(gau, "texture_shadow_spatial_min_resolution", 4)))
    max_res = max(min_res, int(getattr(gau, "texture_shadow_spatial_max_resolution", 32)))
    sigma_factor = max(float(getattr(gau, "texture_sigma_factor", 3.0)), 1e-6)

    scales = gau.get_scaling.detach()
    if scales.ndim != 2 or scales.shape[0] == 0:
        resolutions = torch.empty(0, dtype=torch.int32, device=gau.get_xyz.device)
    else:
        scale_xy = scales[:, :2] if scales.shape[1] >= 2 else scales
        diameter = 2.0 * sigma_factor * scale_xy.max(dim=1).values
        res_float = torch.ceil(diameter / texel_size).clamp(min=min_res, max=max_res)
        if bool(getattr(gau, "texture_shadow_spatial_power2", True)):
            res_float = torch.pow(2.0, torch.ceil(torch.log2(res_float.clamp_min(1.0))))
            res_float = res_float.clamp(min=min_res, max=max_res)
        resolutions = res_float.to(torch.int32)
    return _build_square_texture_dims(resolutions)


def _shadow_fill_distances(resolution, device):
    key = (int(resolution), str(device))
    cached = _SHADOW_FILL_DISTANCE_CACHE.get(key)
    if cached is not None and cached.device == device:
        return cached
    coords_y, coords_x = torch.meshgrid(
        torch.arange(resolution, device=device, dtype=torch.float32),
        torch.arange(resolution, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack([coords_y.reshape(-1), coords_x.reshape(-1)], dim=1)
    dist2 = torch.cdist(coords, coords, p=2).pow(2)
    _SHADOW_FILL_DISTANCE_CACHE[key] = dist2
    return dist2


def _fill_shadow_holes_nearest(maps, hit_weights, chunk_size):
    valid = hit_weights > 1e-8
    if bool(valid.all()):
        return maps
    batch, _, resolution, _ = maps.shape
    texels = resolution * resolution
    flat_maps = maps.reshape(batch, texels)
    flat_valid = valid.reshape(batch, texels)
    has_any = flat_valid.any(dim=1, keepdim=True)
    if not bool(has_any.any()):
        return maps

    chunk_size = max(1, int(chunk_size))
    dist2 = _shadow_fill_distances(resolution, maps.device).to(dtype=maps.dtype)
    filled = flat_maps.clone()
    large = torch.tensor(1e20, dtype=maps.dtype, device=maps.device)
    for start in range(0, batch, chunk_size):
        end = min(start + chunk_size, batch)
        valid_chunk = flat_valid[start:end]
        if not bool(valid_chunk.any()):
            continue
        score = dist2.unsqueeze(0) + torch.where(
            valid_chunk[:, None, :],
            torch.zeros((), dtype=maps.dtype, device=maps.device),
            large,
        )
        nearest = torch.argmin(score, dim=2)
        nearest_values = torch.gather(flat_maps[start:end], 1, nearest)
        fill_mask = (~valid_chunk) & has_any[start:end]
        filled[start:end] = torch.where(fill_mask, nearest_values, filled[start:end])
    return filled.reshape_as(maps)


def _fill_shadow_holes_local(maps, hit_weights):
    valid = (hit_weights > 1e-8).to(dtype=maps.dtype)
    if bool((valid > 0.5).all()):
        return maps
    kernel = torch.tensor(
        [[0.70710678, 1.0, 0.70710678],
         [1.0, 0.0, 1.0],
         [0.70710678, 1.0, 0.70710678]],
        dtype=maps.dtype,
        device=maps.device,
    ).reshape(1, 1, 3, 3)
    neighbor_weight = F.conv2d(valid, kernel, padding=1)
    neighbor_value = F.conv2d(maps * valid, kernel, padding=1) / torch.clamp_min(neighbor_weight, 1e-6)
    fill_mask = (valid <= 0.5) & (neighbor_weight > 0.0)
    return torch.where(fill_mask, neighbor_value, maps)


def _spatial_shadow_to_static(out_trans, non_trans, shadow_dims, base_resolution, gau, return_raw=False):
    device = out_trans.device
    num_points = int(shadow_dims.shape[0])
    out_shadow = torch.empty((num_points, base_resolution, base_resolution), dtype=out_trans.dtype, device=device)
    out_conf = torch.empty_like(out_shadow)
    out_raw_shadow = torch.empty_like(out_shadow) if return_raw else None

    mode = str(getattr(gau, "texture_shadow_hole_fill", "none")).lower()
    if mode not in {"none", "local", "nearest"}:
        raise ValueError(f"Unknown texture_shadow_hole_fill: {mode}")
    chunk_size = int(getattr(gau, "texture_shadow_hole_fill_chunk", 16))

    resolutions = shadow_dims[:, 0].to(torch.long)
    offsets = shadow_dims[:, 2].to(torch.long)
    for resolution_tensor in torch.unique(resolutions, sorted=True):
        resolution = int(resolution_tensor.item())
        mask = resolutions == resolution_tensor
        point_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
        starts = offsets[point_ids]
        local = torch.arange(resolution * resolution, dtype=torch.long, device=device)
        flat_ids = starts[:, None] + local[None, :]
        trans_maps = out_trans[flat_ids].reshape(-1, 1, resolution, resolution)
        weight_maps = non_trans[flat_ids].reshape(-1, 1, resolution, resolution)
        shadow_maps = trans_maps / torch.clamp_min(weight_maps, 1e-6)
        raw_shadow_maps = shadow_maps
        if mode == "local":
            shadow_maps = _fill_shadow_holes_local(shadow_maps, weight_maps)
        elif mode == "nearest":
            shadow_maps = _fill_shadow_holes_nearest(shadow_maps, weight_maps, chunk_size)
        if resolution != base_resolution:
            if return_raw:
                raw_shadow_maps = F.interpolate(raw_shadow_maps, size=(base_resolution, base_resolution), mode="area")
            shadow_maps = F.interpolate(shadow_maps, size=(base_resolution, base_resolution), mode="area")
            weight_maps = F.interpolate(weight_maps, size=(base_resolution, base_resolution), mode="area")
        out_shadow[point_ids] = shadow_maps[:, 0]
        if return_raw:
            out_raw_shadow[point_ids] = raw_shadow_maps[:, 0]
        mean_weight = torch.clamp_min(weight_maps.flatten(1).mean(dim=1, keepdim=True), 1e-6)
        out_conf[point_ids] = torch.clamp(
            weight_maps[:, 0] / mean_weight.reshape(-1, 1, 1),
            0.0,
            1.0,
        )
    if return_raw:
        return out_shadow, out_conf, out_raw_shadow
    return out_shadow, out_conf


def _static_shadow_to_dynamic_flat(static_maps, texture_dims):
    if texture_dims.numel() == 0:
        return static_maps.reshape(-1)
    device = static_maps.device
    dtype = static_maps.dtype
    num_points = int(texture_dims.shape[0])
    total_texels = int((texture_dims[:, 0].to(torch.long) * texture_dims[:, 1].to(torch.long)).sum().item())
    out = torch.empty((total_texels,), dtype=dtype, device=device)
    widths = texture_dims[:, 0].to(torch.long)
    heights = texture_dims[:, 1].to(torch.long)
    offsets = texture_dims[:, 2].to(torch.long)
    base_h = int(static_maps.shape[-2])
    base_w = int(static_maps.shape[-1])
    for wh in torch.unique(torch.stack([widths, heights], dim=1), dim=0):
        width = int(wh[0].item())
        height = int(wh[1].item())
        mask = (widths == wh[0]) & (heights == wh[1])
        point_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
        maps = static_maps[point_ids].reshape(-1, 1, base_h, base_w)
        if height != base_h or width != base_w:
            maps = F.interpolate(maps, size=(height, width), mode="area")
        local = torch.arange(height * width, dtype=torch.long, device=device)
        flat_ids = offsets[point_ids, None] + local[None, :]
        out[flat_ids.reshape(-1)] = maps[:, 0].reshape(-1)
    return out


_TEXTURE_EFFECT_DEFAULT = "uvshadow_specular_lobe"
_TEXTURE_EFFECT_DEFAULT_ALIASES = {
    "uvshadow_specular_residual",
    "uvshadow_specular_lobe",
    "uvshadow_roughness_residual",
    "uvshadow_specular_tint",
    "uvshadow_ks_tint",
    "microtex",
    "uv_specular_gain",
}


def _pack_deferred_texture_mbrdf(basecolor, shadow, other_effects_g, dist_2_inv, num_points):
    """Pack per-UV RGB+shadow into texture, with Gaussian other_effects as channels 4:7."""
    texture_color = torch.cat([basecolor, shadow], dim=1).contiguous()
    colors_precomp = torch.zeros((num_points, 7), dtype=texture_color.dtype, device=texture_color.device)
    colors_precomp[:, 4:7] = other_effects_g * dist_2_inv
    return {
        "texture_color": texture_color,
        "colors_precomp": colors_precomp,
        "basecolor": basecolor,
        "shadow": shadow,
        "other_effects": colors_precomp[:, 4:7],
    }


def _texture_tor_is_active(gau, iteration):
    return (
        torch.is_grad_enabled()
        and bool(getattr(gau, "texture_tor_enabled", False))
        and int(iteration) >= int(getattr(gau, "texture_tor_start_iter", 30_000))
    )


def _texture_tor_gate(signal, gau):
    floor = max(0.0, min(float(getattr(gau, "texture_tor_gate_floor", 0.05)), 1.0))
    scale = max(float(getattr(gau, "texture_tor_gate_scale", 1.0)), 1e-6)
    strength = max(0.0, min(float(getattr(gau, "texture_tor_strength", 1.0)), 1.0))
    signal = torch.nan_to_num(torch.abs(signal.detach()), nan=0.0, posinf=0.0, neginf=0.0)
    gate = signal / (signal.mean().clamp_min(1e-8) * scale)
    gate = torch.clamp(gate, min=floor, max=1.0)
    if strength < 1.0:
        gate = 1.0 + strength * (gate - 1.0)
    return gate.detach()


def _texture_tor_route(value, signal, gau):
    gate = _texture_tor_gate(signal, gau)
    return value.detach() + gate * (value - value.detach())


def _texture_shadow_confidence_is_active(gau, iteration):
    return (
        bool(getattr(gau, "texture_shadow_confidence_enabled", False))
        and int(iteration) >= int(getattr(gau, "texture_shadow_confidence_start_iter", 30_000))
    )


def _reshape_shadow_confidence_like(confidence, reference):
    if confidence.shape == reference.shape:
        return confidence
    if reference.ndim == 4 and confidence.ndim == 3:
        return confidence.unsqueeze(1)
    if reference.ndim == 2 and confidence.ndim == 1:
        return confidence.unsqueeze(-1)
    return confidence.reshape_as(reference)


def _apply_shadow_confidence_residual(shadow_delta, confidence, gau, iteration, counts=None, texel_ids=None):
    if confidence is None or not _texture_shadow_confidence_is_active(gau, iteration):
        return shadow_delta

    strength = max(0.0, min(float(getattr(gau, "texture_shadow_confidence_strength", 1.0)), 1.0))
    if strength <= 0.0:
        return shadow_delta

    confidence = _reshape_shadow_confidence_like(confidence, shadow_delta).to(
        device=shadow_delta.device,
        dtype=shadow_delta.dtype,
    )
    confidence = torch.nan_to_num(confidence.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    gamma = max(float(getattr(gau, "texture_shadow_confidence_gamma", 1.0)), 1e-6)
    if abs(gamma - 1.0) > 1e-6:
        confidence = confidence.pow(gamma)

    routed = shadow_delta * confidence
    if bool(getattr(gau, "texture_shadow_confidence_zero_dc", False)):
        if routed.ndim == 4:
            routed = routed - routed.mean(dim=(2, 3), keepdim=True)
        elif counts is not None and texel_ids is not None:
            values = routed.reshape(routed.shape[0], -1)
            accum = torch.zeros((counts.shape[0], values.shape[1]), dtype=values.dtype, device=values.device)
            accum.index_add_(0, texel_ids, values)
            mean = accum / counts.to(values.dtype).unsqueeze(-1).clamp_min(1.0)
            routed = (values - mean[texel_ids]).reshape_as(routed)
        else:
            routed = routed - routed.mean()

    return shadow_delta + strength * (routed - shadow_delta)


def _compose_shadow_transport(decay, per_uv_shadow, point_shadow, gau, iteration, confidence=None, counts=None, texel_ids=None):
    mode = str(getattr(gau, "texture_shadow_transport_mode", "additive")).lower()
    eps = max(float(getattr(gau, "texture_shadow_transport_eps", 1e-4)), 1e-8)
    if mode == "additive":
        shadow_delta = per_uv_shadow - point_shadow
        shadow_delta = _apply_shadow_confidence_residual(
            shadow_delta,
            confidence,
            gau,
            iteration,
            counts=counts,
            texel_ids=texel_ids,
        )
        return torch.clamp(decay + shadow_delta, 0.0, 1.0)

    uv = torch.clamp(per_uv_shadow, eps, 1.0 - eps)
    point = torch.clamp(point_shadow, eps, 1.0 - eps)
    decay = torch.clamp(decay, eps, 1.0 - eps)
    if mode in {"logit_relative", "log_relative", "lrvr"}:
        return torch.sigmoid(torch.logit(decay) + torch.logit(uv) - torch.logit(point))
    if mode in {"ratio", "relative"}:
        return torch.clamp(decay * uv / point, 0.0, 1.0)
    raise ValueError(f"Unknown texture_shadow_transport_mode: {mode}")


_SURGERY_NONE = {"", "none", "off", "disabled", "full"}


def _texture_factor_surgery_mode(gau):
    return str(getattr(gau, "texture_factor_surgery", "none")).lower()


def _texture_factor_surgery_seed(gau):
    return int(getattr(gau, "texture_factor_surgery_seed", 0))


def _chart_mean_static(chart):
    if chart.ndim < 4:
        return chart
    return chart.mean(dim=(-2, -1), keepdim=True).expand_as(chart)


def _chart_shuffle_static(chart, seed=0):
    if chart.ndim < 4:
        return chart
    texels = chart.shape[-2] * chart.shape[-1]
    if texels <= 1:
        return chart
    perm = torch.arange(texels - 1, -1, -1, device=chart.device, dtype=torch.long)
    shift = int(seed) % texels
    if shift:
        perm = torch.roll(perm, shifts=shift, dims=0)
    return chart.flatten(-2)[..., perm].view_as(chart)


def _apply_factor_surgery_chart(chart, mode, factor, seed=0):
    mean_modes = {f"mean_{factor}", f"{factor}_mean"}
    shuffle_modes = {f"shuffle_{factor}", f"{factor}_shuffle"}
    if factor == "kd":
        mean_modes.update({"mean_albedo", "albedo_mean", "no_kd_chart", "no_albedo_chart"})
        shuffle_modes.update({"shuffle_albedo", "albedo_shuffle"})
    elif factor == "visibility":
        mean_modes.update({"no_v_residual", "no_visibility_residual", "mean_shadow", "no_shadow_chart"})
        shuffle_modes.update({"shuffle_v_residual", "shuffle_visibility_residual", "shuffle_shadow"})
    elif factor in {"gain", "lobe"}:
        mean_modes.update({"mean_specular", "mean_specular_chart"})
        shuffle_modes.update({"shuffle_specular", "shuffle_specular_chart"})
    if mode in mean_modes:
        return _chart_mean_static(chart)
    if mode in shuffle_modes or mode == "shuffle_all":
        return _chart_shuffle_static(chart, seed=seed)
    return chart


def _apply_factor_surgery_unit(chart, mode, factor):
    unit_modes = {
        "gain": {"no_gain", "unit_gain", "no_specular_chart", "no_gain_lobe", "no_specular_factors"},
        "lobe": {"no_lobe", "unit_lobe", "no_specular_chart", "no_gain_lobe", "no_specular_factors"},
    }
    if mode in unit_modes.get(factor, set()):
        return torch.ones_like(chart)
    return chart


def _look_at_2dgs(camera_position, target_position, up_dir):
    camera_direction = camera_position - target_position
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    if abs(np.dot(up_dir, camera_direction)) > 0.9:
        up_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    camera_right = np.cross(up_dir, camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)

    rotation_transform = np.zeros((4, 4), dtype=np.float32)
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[3, 3] = 1.0

    translation_transform = np.eye(4, dtype=np.float32)
    translation_transform[:3, -1] = -np.asarray(camera_position, dtype=np.float32)

    look_at_transform = rotation_transform @ translation_transform
    look_at_transform[1:3, :] *= -1
    return look_at_transform.T


def _build_light_transform_2dgs(viewpoint_camera, means3d, pipe):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx_origin = viewpoint_camera.image_width / (2.0 * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2.0 * tanfovy)

    object_center = means3d.mean(dim=0).detach().cpu().numpy()
    light_position = viewpoint_camera.pl_pos.detach().cpu().numpy().reshape(-1)
    world_view_transform_light = _look_at_2dgs(
        light_position,
        object_center,
        up_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    world_view_transform_light = torch.tensor(
        world_view_transform_light,
        device=viewpoint_camera.world_view_transform.device,
        dtype=viewpoint_camera.world_view_transform.dtype,
    )

    camera_position = viewpoint_camera.camera_center.detach().cpu().numpy() * getattr(pipe, "shadow_light_scale", 1.0)
    light_norm = max(float(np.sum(light_position * light_position)), 1e-8)
    camera_norm = max(float(np.sum(camera_position * camera_position)), 1e-8)
    f_scale_ratio = math.sqrt(light_norm / camera_norm)
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio

    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far
    resolution_scale = max(float(getattr(pipe, "shadow_resolution_scale", 1.0)), 0.25)
    h_light = min(max(32, int(resolution_scale * viewpoint_camera.image_height)), 2048)
    w_light = min(max(32, int(resolution_scale * viewpoint_camera.image_width)), 2048)

    light_persp_proj_matrix = getProjectionMatrix(
        znear=viewpoint_camera.znear,
        zfar=viewpoint_camera.zfar,
        fovX=2.0 * math.atan(tanfovx_far),
        fovY=2.0 * math.atan(tanfovy_far),
    ).transpose(0, 1).cuda()
    light_projmatrix = (
        world_view_transform_light.unsqueeze(0).bmm(light_persp_proj_matrix.unsqueeze(0))
    ).squeeze(0)

    return dict(
        world_view_transform_light=world_view_transform_light,
        light_projmatrix=light_projmatrix,
        tanfovx_far=tanfovx_far,
        tanfovy_far=tanfovy_far,
        h_light=h_light,
        w_light=w_light,
        light_position=light_position,
    )


def _compute_texture_shadow_pass(viewpoint_camera, gau, pipe, bg_color, scaling_modifier=1.0, per_uv=True):
    if viewpoint_camera.pl_pos is None or gau.get_xyz.numel() == 0:
        return None

    means3d = gau.get_xyz
    lt = _build_light_transform_2dgs(viewpoint_camera, means3d, pipe)
    dynamic_textures = bool(getattr(gau, "has_dynamic_textures", False))
    use_per_uv_shadow = bool(per_uv)
    spatial_shadow = use_per_uv_shadow and _texture_shadow_spatial_enabled(gau)
    texture_dims = (
        _build_spatial_shadow_dims(gau)
        if spatial_shadow
        else (
            gau.get_texture_dims
            if dynamic_textures
            else torch.empty(0, device="cuda", dtype=torch.int32)
        )
    )

    shadow_settings = ShadowRasterSettings(
        image_height=lt["h_light"],
        image_width=lt["w_light"],
        tanfovx=lt["tanfovx_far"],
        tanfovy=lt["tanfovy_far"],
        bg=bg_color[:3],
        scale_modifier=scaling_modifier,
        viewmatrix=lt["world_view_transform_light"],
        projmatrix=lt["light_projmatrix"],
        sh_degree=gau.active_sh_degree,
        campos=torch.tensor(lt["light_position"], dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
        low_pass_filter_radius=0.3,
        ortho=False,
        use_textures=use_per_uv_shadow,
        texture_shadow_use_alpha=bool(getattr(pipe, "texture_shadow_use_alpha", False)),
        texture_shadow_output_uv=bool(getattr(pipe, "texture_shadow_output_uv", True)),
        texture_shadow_alpha_bilinear=bool(getattr(pipe, "texture_shadow_alpha_bilinear", False)),
    )
    shadow_rasterizer = ShadowRasterizer(raster_settings=shadow_settings)
    light_colors = torch.ones((means3d.shape[0], 3), dtype=torch.float32, device="cuda")
    texture_shadow_uses_alpha = bool(getattr(pipe, "texture_shadow_use_alpha", False))
    if use_per_uv_shadow:
        # The shadow kernel needs a texture-shaped tensor to size/index per-UV
        # buffers even when alpha modulation is disabled. In that default path
        # the CUDA code never reads alpha values, so avoid a full sigmoid over
        # the atlas just to provide layout metadata.
        if spatial_shadow:
            if texture_shadow_uses_alpha:
                raise RuntimeError("texture_shadow_use_alpha is not supported with spatial shadow resolution yet.")
            total_shadow_texels = int((texture_dims[:, 0].to(torch.long) * texture_dims[:, 1].to(torch.long)).sum().item())
            texture_alpha = torch.empty((total_shadow_texels, 1), dtype=torch.float32, device="cuda")
        else:
            texture_alpha = gau.get_texture_alpha if texture_shadow_uses_alpha else getattr(gau, "_tex_alpha", torch.empty(0, device="cuda"))
    else:
        texture_alpha = torch.empty(0, device="cuda")

    _, _, _, out_trans, non_trans, _ = shadow_rasterizer(
        means3D=means3d,
        means2D=torch.zeros_like(means3d, requires_grad=True),
        shs=None,
        colors_precomp=light_colors,
        opacities=gau.get_opacity,
        scales=gau.get_scaling,
        rotations=gau.get_rotation,
        cov3Ds_precomp=None,
        texture_alpha=texture_alpha,
        texture_dims=texture_dims if use_per_uv_shadow else torch.empty(0, device="cuda", dtype=torch.int32),
        texture_sigma_factor=float(getattr(gau, "texture_sigma_factor", 3.0)),
        non_trans=None,
        offset=getattr(pipe, "shadow_offset", 0.015),
        thres=-1.0,
        is_train=False,
    )

    non_trans_safe = torch.clamp_min(non_trans, 1e-6)
    shadow_output_uv = bool(getattr(pipe, "texture_shadow_output_uv", True))
    if spatial_shadow and shadow_output_uv:
        tex_res = int(getattr(gau, "texture_resolution", 1))
        need_unfilled_shadow_for_rtd = (
            dynamic_textures
            and bool(getattr(gau, "texture_rtd_enabled", False))
            and bool(getattr(gau, "_texture_rtd_collect_shadow_now", False))
            and str(getattr(gau, "texture_shadow_hole_fill", "none")).lower() != "none"
        )
        shadow_static_result = _spatial_shadow_to_static(
            out_trans.reshape(-1),
            non_trans.reshape(-1),
            texture_dims,
            tex_res,
            gau,
            return_raw=need_unfilled_shadow_for_rtd,
        )
        if need_unfilled_shadow_for_rtd:
            per_uv_shadow_static, per_uv_confidence_static, per_uv_shadow_static_rtd = shadow_static_result
        else:
            per_uv_shadow_static, per_uv_confidence_static = shadow_static_result
            per_uv_shadow_static_rtd = per_uv_shadow_static
        point_out = _cached_sum_dynamic_texture_values(gau, out_trans.reshape(-1), texture_dims)
        point_non = _cached_sum_dynamic_texture_values(gau, non_trans.reshape(-1), texture_dims)
        per_point_shadow = (point_out / torch.clamp_min(point_non, 1e-6)).unsqueeze(-1)
        if dynamic_textures:
            per_uv_shadow = _static_shadow_to_dynamic_flat(per_uv_shadow_static, gau.get_texture_dims)
            per_uv_shadow_rtd = _static_shadow_to_dynamic_flat(per_uv_shadow_static_rtd, gau.get_texture_dims)
            per_uv_confidence = _static_shadow_to_dynamic_flat(per_uv_confidence_static, gau.get_texture_dims)
        else:
            per_uv_shadow = per_uv_shadow_static
            per_uv_shadow_rtd = per_uv_shadow_static_rtd
            per_uv_confidence = per_uv_confidence_static
    elif not use_per_uv_shadow or not shadow_output_uv:
        per_point_shadow = (out_trans / non_trans_safe).unsqueeze(-1)
        if bool(getattr(gau, "has_dynamic_textures", False)):
            dims = gau.get_texture_dims
            counts = (dims[:, 0].to(torch.long) * dims[:, 1].to(torch.long)).clamp_min(1)
            per_uv_shadow = torch.repeat_interleave(per_point_shadow.reshape(-1), counts)
            per_uv_shadow_rtd = per_uv_shadow
            per_uv_confidence = torch.ones_like(per_uv_shadow)
        else:
            tex_res = int(getattr(gau, "texture_resolution", 1))
            per_uv_shadow = per_point_shadow.reshape(means3d.shape[0], 1, 1).expand(means3d.shape[0], tex_res, tex_res)
            per_uv_shadow_rtd = per_uv_shadow
            per_uv_confidence = torch.ones_like(per_uv_shadow)
    elif texture_dims.numel() > 0:
        per_uv_shadow = (out_trans / non_trans_safe).reshape(-1)
        per_uv_shadow_rtd = per_uv_shadow
        point_out = _cached_sum_dynamic_texture_values(gau, out_trans.reshape(-1), texture_dims)
        point_non = _cached_sum_dynamic_texture_values(gau, non_trans.reshape(-1), texture_dims)
        per_point_shadow = (point_out / torch.clamp_min(point_non, 1e-6)).unsqueeze(-1)
        counts, texel_ids, _, _ = _cached_dynamic_texture_flat_ids(
            gau, texture_dims, non_trans.device
        )
        mean_non = torch.clamp_min(
            point_non.reshape(-1, 1) / counts.to(point_non.dtype).reshape(-1, 1),
            1e-6,
        )
        per_uv_confidence = torch.clamp(non_trans.reshape(-1, 1) / mean_non[texel_ids], 0.0, 1.0).reshape(-1)
    else:
        tex_res = int(getattr(gau, "texture_resolution", 1))
        per_uv_shadow = (out_trans / non_trans_safe).view(means3d.shape[0], tex_res, tex_res)
        per_uv_shadow_rtd = per_uv_shadow
        point_out = out_trans.reshape(means3d.shape[0], -1).sum(dim=1, keepdim=True)
        point_non = non_trans.reshape(means3d.shape[0], -1).sum(dim=1, keepdim=True)
        per_point_shadow = point_out / torch.clamp_min(point_non, 1e-6)
        non_flat = non_trans.reshape(means3d.shape[0], -1)
        mean_non = torch.clamp_min(non_flat.mean(dim=1, keepdim=True), 1e-6)
        per_uv_confidence = torch.clamp(non_flat / mean_non, 0.0, 1.0).view(means3d.shape[0], tex_res, tex_res)

    return {
        "per_point_shadow": per_point_shadow,
        "per_uv_shadow": per_uv_shadow,
        "per_uv_shadow_rtd": per_uv_shadow_rtd,
        "per_uv_confidence": per_uv_confidence,
        "light_viewmatrix": lt["world_view_transform_light"],
        "light_projmatrix": lt["light_projmatrix"],
        "spatial_shadow_dims": texture_dims if spatial_shadow else None,
    }


def _compute_texture_mbrdf(viewpoint_camera, gau, shadow_pkg, fix_labert=False, iteration=0):
    if viewpoint_camera.pl_pos is None:
        return None

    dev = gau.get_xyz.device
    num_points = gau.get_xyz.shape[0]
    pl_pos = viewpoint_camera.pl_pos
    if not isinstance(pl_pos, torch.Tensor):
        pl_pos = torch.tensor(pl_pos, dtype=torch.float32)
    pl_pos = pl_pos.to(dev)
    if pl_pos.ndim == 1:
        pl_pos = pl_pos.unsqueeze(0)

    cam_center = viewpoint_camera.camera_center
    if not isinstance(cam_center, torch.Tensor):
        cam_center = torch.tensor(cam_center, dtype=torch.float32)
    cam_center = cam_center.to(dev)
    if os.getenv("GS3_2DGS_DETACH_VIEWDIR", "0") == "1":
        cam_center = cam_center.detach()

    pl_pos3 = pl_pos[0].unsqueeze(0).expand(num_points, -1)
    wi_ray = pl_pos3 - gau.get_xyz
    wi_dist2 = wi_ray.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
    dist_2_inv = 1.0 / wi_dist2
    wi = wi_ray * dist_2_inv.sqrt()
    wo = _safe_normalize(cam_center - gau.get_xyz)

    local_axises = gau.get_local_axis
    local_z = local_axises[:, :, 2]
    wi_local = _project_to_local(wi, local_axises)
    wo_local = _project_to_local(wo, local_axises)
    cos_theta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)
    asg_scales = gau.asg_func.get_asg_lam_miu
    asg_axises = gau.asg_func.get_asg_axis
    asg_1 = gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises)

    per_uv_shadow = None if shadow_pkg is None else shadow_pkg.get("per_uv_shadow")
    if per_uv_shadow is None:
        raise RuntimeError("Texture rendering requires the texture-aware shadow pass.")

    if bool(getattr(gau, "has_dynamic_textures", False)):
        texture_effect_mode = str(getattr(gau, "texture_effect_mode", _TEXTURE_EFFECT_DEFAULT))

        texture_dims = gau.get_texture_dims
        counts, texel_ids, flat_ids, total_texels = _cached_dynamic_texture_flat_ids(gau, texture_dims, dev)
        per_uv_shadow = per_uv_shadow.reshape(total_texels, 1)

        if texture_effect_mode in _TEXTURE_EFFECT_DEFAULT_ALIASES:
            use_specular_tint = bool(getattr(gau, "_uses_texture_specular_tint", lambda: False)())
            use_specular_lobe = bool(getattr(gau, "_uses_texture_specular_lobe", lambda: False)())
            point_shadow = _require_point_shadow(shadow_pkg)
            decay_g, other_effects_g, _, _ = gau.neural_phasefunc(
                wi,
                wo,
                gau.get_xyz,
                gau.get_neural_material,
                hint=point_shadow,
            )
            if decay_g is None:
                decay_g = torch.ones((num_points, 1), dtype=torch.float32, device=dev)
            if other_effects_g is None:
                other_effects_g = torch.zeros((num_points, 3), dtype=torch.float32, device=dev)

            texture_kd = gau.get_texture_color
            specular_gain = gau.get_texture_specular_gain
            if specular_gain.numel() == 0:
                specular_gain = torch.ones((total_texels, 1), dtype=torch.float32, device=dev)
            basecolor = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
            shadow = torch.empty((total_texels, 1), dtype=torch.float32, device=dev)
            shadow[:] = _compose_shadow_transport(
                decay_g[texel_ids],
                per_uv_shadow,
                point_shadow[texel_ids],
                gau,
                iteration,
                shadow_pkg.get("per_uv_confidence"),
                counts=counts,
                texel_ids=texel_ids,
            )
            tor_active = _texture_tor_is_active(gau, iteration)
            direct_signal = shadow.detach() * cos_theta[texel_ids].detach() * dist_2_inv[texel_ids].detach()
            diffuse_flat = texture_kd / math.pi
            if tor_active:
                diffuse_flat = _texture_tor_route(diffuse_flat, direct_signal, gau)
            if fix_labert:
                specular_flat = 0.0
            elif use_specular_tint:
                specular_tint = gau.get_texture_specular_tint
                if specular_tint.numel() == 0:
                    specular_tint = gau.get_ks[texel_ids]
                specular_flat = specular_tint * asg_1[texel_ids]
            elif use_specular_lobe:
                lobe_scale = gau.get_texture_specular_lobe_scale
                if lobe_scale.numel() == 0:
                    lobe_scale = torch.ones((total_texels, 1), dtype=torch.float32, device=dev)
                asg_clamped = torch.clamp(asg_1[texel_ids], min=1e-8)
                if tor_active:
                    ks_signal = gau.get_ks[texel_ids].detach().abs().mean(dim=1, keepdim=True)
                    asg_signal = asg_clamped.detach().abs().mean(dim=1, keepdim=True)
                    spec_signal = direct_signal * ks_signal * asg_signal
                    specular_gain = _texture_tor_route(specular_gain, spec_signal, gau)
                    lobe_signal = spec_signal * torch.abs(torch.log(asg_clamped.detach())).mean(dim=1, keepdim=True)
                    lobe_scale = _texture_tor_route(lobe_scale, lobe_signal, gau)
                asg_lobed = asg_clamped.pow(lobe_scale)
                specular_flat = gau.get_ks[texel_ids] * asg_lobed * specular_gain
            else:
                if tor_active:
                    ks_signal = gau.get_ks[texel_ids].detach().abs().mean(dim=1, keepdim=True)
                    asg_signal = asg_1[texel_ids].detach().abs().mean(dim=1, keepdim=True)
                    specular_gain = _texture_tor_route(specular_gain, direct_signal * ks_signal * asg_signal, gau)
                specular_flat = gau.get_ks[texel_ids] * asg_1[texel_ids] * specular_gain
            basecolor[:] = (
                (diffuse_flat + specular_flat)
                * cos_theta[texel_ids]
                * dist_2_inv[texel_ids]
            )
            return _pack_deferred_texture_mbrdf(basecolor, shadow, other_effects_g, dist_2_inv, num_points)

        if texture_effect_mode == "per_uv_micro_normal":
            basecolor = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
            shadow = torch.empty((total_texels, 1), dtype=torch.float32, device=dev)
            other_effects = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
            texture_kd = gau.get_texture_color
            texture_local_q = gau.get_texture_local_q
            if texture_local_q.numel() == 0:
                texture_local_q = None

            local_axes_uv = _frame_from_texture_local_q(
                local_axises[texel_ids],
                texture_local_q[flat_ids] if texture_local_q is not None else None,
            )
            wi_uv = wi[texel_ids]
            wo_uv = wo[texel_ids]
            wi_local_uv = _project_to_local(wi_uv, local_axes_uv)
            wo_local_uv = _project_to_local(wo_uv, local_axes_uv)
            asg_uv = gau.asg_func(
                wi_local_uv,
                wo_local_uv,
                gau.get_alpha_asg[texel_ids],
                asg_scales,
                asg_axises,
            )

            decay_flat, other_effects_flat, _, _ = gau.neural_phasefunc(
                wi_uv,
                wo_uv,
                gau.get_xyz[texel_ids],
                gau.get_neural_material[texel_ids],
                hint=per_uv_shadow[flat_ids],
            )
            if decay_flat is None:
                decay_flat = torch.ones((total_texels, 1), dtype=torch.float32, device=dev)
            if other_effects_flat is None:
                other_effects[:] = 0.0
            else:
                other_effects[flat_ids] = other_effects_flat * dist_2_inv[texel_ids]

            diffuse_flat = texture_kd[flat_ids] / math.pi
            if fix_labert:
                specular_flat = 0.0
            else:
                specular_flat = gau.get_ks[texel_ids] * asg_uv
            basecolor[flat_ids] = (
                (diffuse_flat + specular_flat)
                * cos_theta[texel_ids]
                * dist_2_inv[texel_ids]
            )
            shadow[flat_ids] = decay_flat

            return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

        if texture_effect_mode in {
            "uvshadow_micro_normal_residual",
            "uvshadow_micro_normal_full",
            "uvshadow_micro_normal_specular_residual",
            "uvshadow_micro_normal_specular_full",
        }:
            point_shadow = _require_point_shadow(shadow_pkg)
            decay_g, other_effects_g, _, _ = gau.neural_phasefunc(
                wi,
                wo,
                gau.get_xyz,
                gau.get_neural_material,
                hint=point_shadow,
            )
            if decay_g is None:
                decay_g = torch.ones((num_points, 1), dtype=torch.float32, device=dev)
            if other_effects_g is None:
                other_effects_g = torch.zeros((num_points, 3), dtype=torch.float32, device=dev)

            basecolor = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
            shadow = torch.empty((total_texels, 1), dtype=torch.float32, device=dev)
            other_effects = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
            texture_kd = gau.get_texture_color
            use_specular_gain = "specular" in texture_effect_mode

            texture_local_q = gau.get_texture_local_q
            if texture_local_q.numel() == 0:
                texture_local_q = None
            local_axes_uv = _frame_from_texture_local_q(
                local_axises[texel_ids],
                texture_local_q[flat_ids] if texture_local_q is not None else None,
            )
            wi_uv = wi[texel_ids]
            wo_uv = wo[texel_ids]
            cos_theta_flat = cos_theta[texel_ids]
            if texture_effect_mode in {"uvshadow_micro_normal_full", "uvshadow_micro_normal_specular_full"}:
                cos_theta_flat = _NdotWi(local_axes_uv[:, :, 2], wi_uv, torch.nn.ELU(alpha=0.01), 0.01)
            if fix_labert:
                specular_flat = 0.0
            else:
                asg_uv = gau.asg_func(
                    _project_to_local(wi_uv, local_axes_uv),
                    _project_to_local(wo_uv, local_axes_uv),
                    gau.get_alpha_asg[texel_ids],
                    asg_scales,
                    asg_axises,
                )
                specular_flat = gau.get_ks[texel_ids] * asg_uv
                if use_specular_gain:
                    specular_gain = gau.get_texture_specular_gain
                    if specular_gain.numel() == 0:
                        specular_gain = torch.ones((total_texels, 1), dtype=torch.float32, device=dev)
                    specular_flat = specular_flat * specular_gain[flat_ids]

            diffuse_flat = texture_kd[flat_ids] / math.pi
            basecolor[flat_ids] = (
                (diffuse_flat + specular_flat)
                * cos_theta_flat
                * dist_2_inv[texel_ids]
            )
            shadow_delta = per_uv_shadow[flat_ids] - point_shadow[texel_ids]
            shadow[flat_ids] = torch.clamp(decay_g[texel_ids] + shadow_delta, 0.0, 1.0)
            other_effects[flat_ids] = other_effects_g[texel_ids] * dist_2_inv[texel_ids]

            return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

        if texture_effect_mode != "per_uv":
            raise ValueError("Dynamic texture resolution currently requires texture_effect_mode='uvshadow_specular_lobe' (default), 'uvshadow_specular_residual', 'uvshadow_specular_tint', 'per_uv', 'per_uv_micro_normal', or a legacy uvshadow_micro_normal mode.")

        basecolor = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
        shadow = torch.empty((total_texels, 1), dtype=torch.float32, device=dev)
        other_effects = torch.empty((total_texels, 3), dtype=torch.float32, device=dev)
        texture_kd = gau.get_texture_color

        decay_flat, other_effects_flat, _, _ = gau.neural_phasefunc(
            wi[texel_ids],
            wo[texel_ids],
            gau.get_xyz[texel_ids],
            gau.get_neural_material[texel_ids],
            hint=per_uv_shadow[flat_ids],
        )
        if decay_flat is None:
            decay_flat = torch.ones((total_texels, 1), dtype=torch.float32, device=dev)
        if other_effects_flat is None:
            other_effects[:] = 0.0
        else:
            other_effects[flat_ids] = other_effects_flat * (1.0 / wi_dist2[texel_ids])

        if fix_labert:
            specular_flat = 0.0
        else:
            specular_flat = gau.get_ks[texel_ids] * asg_1[texel_ids]
        diffuse_flat = texture_kd[flat_ids] / math.pi
        shadow[flat_ids] = decay_flat
        basecolor[flat_ids] = (
            (diffuse_flat + specular_flat)
            * cos_theta[texel_ids]
            * dist_2_inv[texel_ids]
        )

        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    tex_res = per_uv_shadow.shape[-1]
    texture_effect_mode = str(getattr(gau, "texture_effect_mode", _TEXTURE_EFFECT_DEFAULT))

    if texture_effect_mode in _TEXTURE_EFFECT_DEFAULT_ALIASES:
        surgery_mode = _texture_factor_surgery_mode(gau)
        surgery_seed = _texture_factor_surgery_seed(gau)
        use_specular_tint = bool(getattr(gau, "_uses_texture_specular_tint", lambda: False)())
        use_specular_lobe = bool(getattr(gau, "_uses_texture_specular_lobe", lambda: False)())
        point_shadow = _require_point_shadow(shadow_pkg)
        decay_g, other_effects_g, _, _ = gau.neural_phasefunc(
            wi,
            wo,
            gau.get_xyz,
            gau.get_neural_material,
            hint=point_shadow,
        )
        if decay_g is None:
            decay_g = torch.ones((num_points, 1), dtype=torch.float32, device=dev)
        if other_effects_g is None:
            other_effects_g = torch.zeros((num_points, 3), dtype=torch.float32, device=dev)

        raw_shadow = per_uv_shadow[:, None, :, :]
        if surgery_mode not in _SURGERY_NONE:
            point_shadow_uv = point_shadow[:, :, None, None].expand_as(raw_shadow)
            if surgery_mode in {"no_v_residual", "no_visibility_residual"}:
                raw_shadow = point_shadow_uv
            elif surgery_mode in {"shuffle_v_residual", "shuffle_visibility_residual"}:
                raw_shadow = point_shadow_uv + _chart_shuffle_static(raw_shadow - point_shadow_uv, seed=surgery_seed)
            else:
                raw_shadow = _apply_factor_surgery_chart(raw_shadow, surgery_mode, "visibility", seed=surgery_seed)
        shadow = _compose_shadow_transport(
            decay_g[:, :, None, None],
            raw_shadow,
            point_shadow[:, :, None, None],
            gau,
            iteration,
            shadow_pkg.get("per_uv_confidence"),
        )
        tor_active = _texture_tor_is_active(gau, iteration)
        direct_signal = shadow.detach() * cos_theta[:, :, None, None].detach() * dist_2_inv[:, :, None, None].detach()
        texture_kd = gau.get_texture_color
        if surgery_mode not in _SURGERY_NONE:
            texture_kd = _apply_factor_surgery_chart(texture_kd, surgery_mode, "kd", seed=surgery_seed)
        texture_diffuse = texture_kd / math.pi
        if tor_active:
            texture_diffuse = _texture_tor_route(texture_diffuse, direct_signal, gau)
        if fix_labert:
            specular_uv = 0.0
        elif use_specular_tint:
            specular_tint = gau.get_texture_specular_tint
            if specular_tint.numel() == 0:
                specular_tint = gau.get_ks[:, :, None, None]
            specular_uv = specular_tint * asg_1[:, :, None, None]
        elif use_specular_lobe:
            specular_gain = gau.get_texture_specular_gain
            if specular_gain.numel() == 0:
                specular_gain = torch.ones((num_points, 1, tex_res, tex_res), dtype=torch.float32, device=dev)
            lobe_scale = gau.get_texture_specular_lobe_scale
            if lobe_scale.numel() == 0:
                lobe_scale = torch.ones((num_points, 1, tex_res, tex_res), dtype=torch.float32, device=dev)
            if surgery_mode not in _SURGERY_NONE:
                specular_gain = _apply_factor_surgery_unit(specular_gain, surgery_mode, "gain")
                specular_gain = _apply_factor_surgery_chart(specular_gain, surgery_mode, "gain", seed=surgery_seed)
                lobe_scale = _apply_factor_surgery_unit(lobe_scale, surgery_mode, "lobe")
                lobe_scale = _apply_factor_surgery_chart(lobe_scale, surgery_mode, "lobe", seed=surgery_seed)
            asg_clamped = torch.clamp(asg_1[:, :, None, None], min=1e-8)
            if tor_active:
                ks_signal = gau.get_ks.detach().abs().mean(dim=1, keepdim=True)[:, :, None, None]
                asg_signal = asg_clamped.detach().abs().mean(dim=1, keepdim=True)
                spec_signal = direct_signal * ks_signal * asg_signal
                specular_gain = _texture_tor_route(specular_gain, spec_signal, gau)
                lobe_signal = spec_signal * torch.abs(torch.log(asg_clamped.detach())).mean(dim=1, keepdim=True)
                lobe_scale = _texture_tor_route(lobe_scale, lobe_signal, gau)
            asg_lobed = asg_clamped.pow(lobe_scale)
            specular_uv = gau.get_ks[:, :, None, None] * asg_lobed * specular_gain
        else:
            specular_gain = gau.get_texture_specular_gain
            if specular_gain.numel() == 0:
                specular_gain = torch.ones((num_points, 1, tex_res, tex_res), dtype=torch.float32, device=dev)
            if surgery_mode not in _SURGERY_NONE:
                specular_gain = _apply_factor_surgery_unit(specular_gain, surgery_mode, "gain")
                specular_gain = _apply_factor_surgery_chart(specular_gain, surgery_mode, "gain", seed=surgery_seed)
            if tor_active:
                ks_signal = gau.get_ks.detach().abs().mean(dim=1, keepdim=True)[:, :, None, None]
                asg_signal = asg_1[:, :, None, None].detach().abs().mean(dim=1, keepdim=True)
                specular_gain = _texture_tor_route(specular_gain, direct_signal * ks_signal * asg_signal, gau)
            specular_uv = (gau.get_ks * asg_1)[:, :, None, None] * specular_gain
        basecolor = (texture_diffuse + specular_uv) * cos_theta[:, :, None, None] * dist_2_inv[:, :, None, None]
        return _pack_deferred_texture_mbrdf(basecolor, shadow, other_effects_g, dist_2_inv, num_points)

    if texture_effect_mode == "per_uv_micro_normal":
        basecolor = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
        shadow = torch.empty((num_points, 1, tex_res, tex_res), dtype=torch.float32, device=dev)
        other_effects = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
        texture_kd = gau.get_texture_color
        texture_local_q = gau.get_texture_local_q
        if texture_local_q.numel() == 0:
            texture_local_q = None

        num_texels = num_points * tex_res * tex_res
        texel_ids = torch.arange(num_points, dtype=torch.long, device=dev).repeat_interleave(tex_res * tex_res)
        q = (
            texture_local_q.permute(0, 2, 3, 1).reshape(num_texels, 4)
            if texture_local_q is not None
            else None
        )
        local_axes_uv = _frame_from_texture_local_q(local_axises[texel_ids], q)
        wi_uv = wi[texel_ids]
        wo_uv = wo[texel_ids]
        wi_local_uv = _project_to_local(wi_uv, local_axes_uv)
        wo_local_uv = _project_to_local(wo_uv, local_axes_uv)
        asg_uv = gau.asg_func(
            wi_local_uv,
            wo_local_uv,
            gau.get_alpha_asg[texel_ids],
            asg_scales,
            asg_axises,
        )

        decay_flat, other_effects_flat, _, _ = gau.neural_phasefunc(
            wi_uv,
            wo_uv,
            gau.get_xyz[texel_ids],
            gau.get_neural_material[texel_ids],
            hint=per_uv_shadow.reshape(num_texels, 1),
        )
        if decay_flat is None:
            decay_flat = torch.ones((num_texels, 1), dtype=torch.float32, device=dev)
        shadow[:] = decay_flat.reshape(num_points, tex_res, tex_res, 1).permute(0, 3, 1, 2).contiguous()
        if other_effects_flat is None:
            other_effects[:] = 0.0
        else:
            other_effects[:] = (
                other_effects_flat * dist_2_inv[texel_ids]
            ).reshape(num_points, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()

        diffuse_flat = (
            texture_kd.permute(0, 2, 3, 1).reshape(num_texels, 3)
            / math.pi
        )
        if fix_labert:
            specular_flat = 0.0
        else:
            specular_flat = gau.get_ks[texel_ids] * asg_uv
        basecolor[:] = (
            (diffuse_flat + specular_flat)
            * cos_theta[texel_ids]
            * dist_2_inv[texel_ids]
        ).reshape(num_points, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()

        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    if texture_effect_mode in {
        "uvshadow_micro_normal_residual",
        "uvshadow_micro_normal_full",
        "uvshadow_micro_normal_specular_residual",
        "uvshadow_micro_normal_specular_full",
    }:
        point_shadow = _require_point_shadow(shadow_pkg)
        decay_g, other_effects_g, _, _ = gau.neural_phasefunc(
            wi,
            wo,
            gau.get_xyz,
            gau.get_neural_material,
            hint=point_shadow,
        )
        if decay_g is None:
            decay_g = torch.ones((num_points, 1), dtype=torch.float32, device=dev)
        if other_effects_g is None:
            other_effects_g = torch.zeros((num_points, 3), dtype=torch.float32, device=dev)

        num_texels = num_points * tex_res * tex_res
        texel_ids = torch.arange(num_points, dtype=torch.long, device=dev).repeat_interleave(tex_res * tex_res)
        texture_local_q = gau.get_texture_local_q
        if texture_local_q.numel() == 0:
            texture_local_q = None
        q = (
            texture_local_q.permute(0, 2, 3, 1).reshape(num_texels, 4)
            if texture_local_q is not None
            else None
        )
        local_axes_uv = _frame_from_texture_local_q(local_axises[texel_ids], q)
        wi_uv = wi[texel_ids]
        wo_uv = wo[texel_ids]
        if texture_effect_mode in {"uvshadow_micro_normal_full", "uvshadow_micro_normal_specular_full"}:
            cos_theta_uv = _NdotWi(local_axes_uv[:, :, 2], wi_uv, torch.nn.ELU(alpha=0.01), 0.01)
            cos_theta_term = cos_theta_uv.reshape(num_points, tex_res, tex_res, 1).permute(
                0, 3, 1, 2
            ).contiguous()
        else:
            cos_theta_term = cos_theta[:, :, None, None]

        texture_diffuse = gau.get_texture_color / math.pi
        if fix_labert:
            specular_uv = 0.0
        else:
            asg_uv = gau.asg_func(
                _project_to_local(wi_uv, local_axes_uv),
                _project_to_local(wo_uv, local_axes_uv),
                gau.get_alpha_asg[texel_ids],
                asg_scales,
                asg_axises,
            )
            specular_uv = (gau.get_ks[texel_ids] * asg_uv).reshape(
                num_points, tex_res, tex_res, 3
            ).permute(0, 3, 1, 2).contiguous()
            if "specular" in texture_effect_mode:
                specular_gain = gau.get_texture_specular_gain
                if specular_gain.numel() == 0:
                    specular_gain = torch.ones((num_points, 1, tex_res, tex_res), dtype=torch.float32, device=dev)
                specular_uv = specular_uv * specular_gain
        raw_shadow = per_uv_shadow[:, None, :, :]
        shadow = torch.clamp(decay_g[:, :, None, None] + raw_shadow - point_shadow[:, :, None, None], 0.0, 1.0)
        other_effects = (other_effects_g * dist_2_inv)[:, :, None, None].expand(-1, -1, tex_res, tex_res)
        basecolor = (texture_diffuse + specular_uv) * cos_theta_term * dist_2_inv[:, :, None, None]
        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    if texture_effect_mode == "uv_specular_gain":
        point_shadow = _require_point_shadow(shadow_pkg)
        decay_g, other_effects_g, _, _ = gau.neural_phasefunc(
            wi,
            wo,
            gau.get_xyz,
            gau.get_neural_material,
            hint=point_shadow,
        )
        if decay_g is None:
            decay_g = torch.ones((num_points, 1), dtype=torch.float32, device=dev)
        if other_effects_g is None:
            other_effects_g = torch.zeros((num_points, 3), dtype=torch.float32, device=dev)

        texture_diffuse = gau.get_texture_color / math.pi
        if fix_labert:
            specular_uv = 0.0
        else:
            specular_gain = gau.get_texture_specular_gain
            if specular_gain.numel() == 0:
                specular_gain = torch.ones((num_points, 1, tex_res, tex_res), dtype=torch.float32, device=dev)
            specular_uv = (gau.get_ks * asg_1)[:, :, None, None] * specular_gain
        raw_shadow = per_uv_shadow[:, None, :, :]
        shadow = torch.clamp(decay_g[:, :, None, None] + raw_shadow - point_shadow[:, :, None, None], 0.0, 1.0)
        other_effects = (other_effects_g * dist_2_inv)[:, :, None, None].expand(-1, -1, tex_res, tex_res)
        basecolor = (texture_diffuse + specular_uv) * cos_theta[:, :, None, None] * dist_2_inv[:, :, None, None]
        return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}

    shadow = torch.empty((num_points, tex_res, tex_res), dtype=torch.float32, device=dev)
    specular_uv = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
    if texture_effect_mode == "per_uv":
        other_effects = torch.empty((num_points, 3, tex_res, tex_res), dtype=torch.float32, device=dev)
    elif texture_effect_mode == "legacy_mean_other":
        other_effects = torch.empty((num_points, 3), dtype=torch.float32, device=dev)
    else:
        raise ValueError(f"Unknown texture_effect_mode: {texture_effect_mode}")

    num_texels = num_points * tex_res * tex_res
    texel_ids = torch.arange(num_points, dtype=torch.long, device=dev).repeat_interleave(tex_res * tex_res)

    def _expand_all(tensor):
        dim = tensor.shape[-1]
        return tensor[:, None, None, :].expand(num_points, tex_res, tex_res, dim).reshape(num_texels, dim)

    decay_flat, other_effects_flat, _, _ = gau.neural_phasefunc(
        _expand_all(wi),
        _expand_all(wo),
        _expand_all(gau.get_xyz),
        _expand_all(gau.get_neural_material),
        hint=per_uv_shadow.reshape(num_texels, 1),
    )
    if decay_flat is None:
        decay_flat = torch.ones((num_texels, 1), dtype=torch.float32, device=dev)

    shadow[:] = decay_flat.reshape(num_points, tex_res, tex_res)
    if fix_labert:
        specular_uv[:] = 0.0
    else:
        specular_uv[:] = (
            _expand_all(gau.get_ks) * _expand_all(asg_1)
        ).reshape(num_points, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()

    if other_effects_flat is None:
        other_effects[:] = 0.0
    else:
        other_effects_uv = (
            other_effects_flat * (1.0 / _expand_all(wi_dist2))
        ).reshape(num_points, tex_res, tex_res, 3).permute(0, 3, 1, 2).contiguous()
        if texture_effect_mode == "per_uv":
            other_effects[:] = other_effects_uv
        else:
            other_effects[:] = other_effects_uv.mean(dim=(2, 3))

    texture_diffuse = gau.get_texture_color / math.pi
    basecolor = (texture_diffuse + specular_uv) * cos_theta[:, :, None, None] * dist_2_inv[:, :, None, None]
    return {"basecolor": basecolor, "shadow": shadow, "other_effects": other_effects}


def _get_shadow_backward_stage(modelset, iteration: int):
    if getattr(modelset, "detach_shadow", False):
        return {"xyz": False, "opacity": False, "scaling": False, "rotation": False}
    if not getattr(modelset, "shadow_backward_stage_enabled", False):
        return {"xyz": True, "opacity": True, "scaling": True, "rotation": True}
    return {
        "xyz": iteration >= int(getattr(modelset, "shadow_backward_xyz_from_iter", 0)),
        "opacity": iteration >= int(getattr(modelset, "shadow_backward_opacity_from_iter", 0)),
        "scaling": iteration >= int(getattr(modelset, "shadow_backward_scaling_from_iter", 0)),
        "rotation": iteration >= int(getattr(modelset, "shadow_backward_rotation_from_iter", 0)),
    }


def _build_render_pkg(render, shadow, other_effects, means2D, radii, allmap, transmat_grad_holder, modelset, iteration, shadow_pkg):
    render_alpha = allmap[1:2].clamp_min(1e-8)
    expected_depth = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    return {
        "render": render,
        "shadow": shadow,
        "other_effects": other_effects,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "out_weight": torch.zeros((means2D.shape[0], 1), dtype=torch.float32, device=means2D.device),
        "backward_info": {},
        "shadow_stage": _get_shadow_backward_stage(modelset, iteration),
        "transmat_grad_holder": transmat_grad_holder,
        "expected_depth": expected_depth,
        "depth_image": expected_depth,
        "pre_shadow": None if shadow_pkg is None else shadow_pkg["per_point_shadow"],
        "texture_shadow_pkg": shadow_pkg,
        "render_base": render,
        "render_shadow": shadow,
        "render_other_effects": other_effects,
    }


def render_2dgs_texture_deferred(
    viewpoint_camera,
    gau,
    pipe,
    bg_color,
    modelset,
    scaling_modifier=1.0,
    fix_labert=False,
    iteration=0,
    light_stream=None,
    calc_stream=None,
):
    if not getattr(gau, "use_MBRDF", False):
        raise RuntimeError("The gs3 texture branch is currently wired for deferred mBRDF training.")

    current_stream = torch.cuda.current_stream()
    light_stream_ctx = torch.cuda.stream(light_stream) if light_stream is not None else nullcontext()
    calc_stream_ctx = torch.cuda.stream(calc_stream) if calc_stream is not None else nullcontext()

    with light_stream_ctx:
        shadow_pkg = _compute_texture_shadow_pass(viewpoint_camera, gau, pipe, bg_color, scaling_modifier)

    if light_stream is not None and calc_stream is not light_stream:
        if calc_stream is not None:
            calc_stream.wait_stream(light_stream)
        else:
            current_stream.wait_stream(light_stream)

    with calc_stream_ctx:
        mbrdf = _compute_texture_mbrdf(viewpoint_camera, gau, shadow_pkg, fix_labert=fix_labert, iteration=iteration)

    if light_stream is not None:
        current_stream.wait_stream(light_stream)
    if calc_stream is not None and calc_stream is not light_stream:
        current_stream.wait_stream(calc_stream)

    rendered_image, radii, allmap, means2D, transmat_grad_holder, rendered_split = rasterize_with_texture_module(
        viewpoint_camera=viewpoint_camera,
        pc=gau,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        inputs=TextureRenderInputs(
            deferred=True,
            mbrdf=mbrdf,
            colors_precomp=None,
            return_split=True,
            compose_deferred=False,
        ),
    )
    render_base = rendered_split[0:3]
    shadow = rendered_split[3:4]
    other_effects = rendered_split[4:7]
    pkg = _build_render_pkg(render_base, shadow, other_effects, means2D, radii, allmap, transmat_grad_holder, modelset, iteration, shadow_pkg)
    if rendered_image is not None:
        pkg["render_composed"] = rendered_image
    return pkg
