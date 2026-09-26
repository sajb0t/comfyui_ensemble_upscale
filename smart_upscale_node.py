# -*- coding: utf-8 -*-
"""
Smart Ensemble Upscale  🔬
==========================

A single ComfyUI custom node that combines three techniques that (to our
knowledge) no existing single ComfyUI upscale node offers together:

  1. Multi-Model Ensemble Blending
     Run 1-3 ESRGAN-style upscale models on the same image and blend the
     results in a *content-aware* way. An edge-map (Sobel) decides, per
     pixel, whether to favour the sharpest / most edge-preserving model
     (in edge regions) or the smoothest model (in flat regions).

  2. Frequency-Separated Upscaling
     Split the image into a low-frequency component (Gaussian blur, carries
     colour / tone / lighting) and a high-frequency component (fine detail).
     The low-frequency part is upscaled with a clean bicubic/lanczos resize
     (models often shift colour), while the crisp detail is taken from the
     neural upscale. Recombining gives sharper detail *without* wrecking the
     colour tones.

  3. Seam-Free Gaussian Tiling
     Tile-based processing (VRAM friendly) where neighbouring tiles overlap
     and are blended with a 2D Gaussian weighting window, so the visible
     seams that plague normal tiled upscalers disappear.

All three are combined in ONE node.

Compatible with ComfyUI's standard IMAGE format: torch.float32 tensors of
shape (B, H, W, C) with values in [0, 1]. Models are chosen by filename and
loaded from ``models/upscale_models``.
"""

import logging
import math
import os
import urllib.request

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional ComfyUI imports.  We degrade gracefully so the file can be linted /
# imported outside of a running ComfyUI instance (e.g. for unit tests).
# ---------------------------------------------------------------------------
try:
    import comfy.model_management as model_management  # type: ignore
    import comfy.utils as comfy_utils  # type: ignore
    import folder_paths  # type: ignore
    from spandrel import ImageModelDescriptor, ModelLoader  # type: ignore

    try:
        from spandrel import MAIN_REGISTRY  # type: ignore
        from spandrel_extra_arches import EXTRA_REGISTRY  # type: ignore
        MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    except ImportError:
        pass

    _HAS_COMFY = True
except Exception:  # pragma: no cover - only hit outside ComfyUI
    model_management = None
    comfy_utils = None
    folder_paths = None
    ImageModelDescriptor = None
    ModelLoader = None
    _HAS_COMFY = False

# Known ESRGAN-style weights. Combo values are treated as untrusted filenames
# and re-checked against this allowlist / folder_paths at load time.
UPSCALE_MODEL_CATALOG = {
    "4x-UltraSharp.pth":
        "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
    "4x-AnimeSharp.pth":
        "https://huggingface.co/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth",
    "RealESRGAN_x4plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "4x_NMKD-Siax_200k.pth":
        "https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth",
    "4x_foolhardy_Remacri.pth":
        "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    "4x_NMKD-Superscale-SP_178000_G.pth":
        "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth",
    "4xNomos8kSC.pth":
        "https://github.com/Phhofm/models/releases/download/4xNomos8kSC/4xNomos8kSC.pth",
    "RealESRGAN_x2plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}
_NONE = "none"
_CUSTOM = "Custom"

# None means the three model dropdowns are used as-is.
MODEL_PRESETS = {
    "Realistic": (
        "4xNomos8kSC.pth",
        "RealESRGAN_x4plus.pth",
        "4x_NMKD-Superscale-SP_178000_G.pth",
    ),
    "Anime": (
        "4x-AnimeSharp.pth",
        "RealESRGAN_x4plus_anime_6B.pth",
        "4x-UltraSharp.pth",
    ),
    "Sharp": (
        "4x-UltraSharp.pth",
        "4x_foolhardy_Remacri.pth",
        "4xNomos8kSC.pth",
    ),
    "Smooth": (
        "4x_NMKD-Superscale-SP_178000_G.pth",
        "4xNomos8kSC.pth",
        "RealESRGAN_x4plus.pth",
    ),
    "2x": (
        "RealESRGAN_x2plus.pth",
        _NONE,
        _NONE,
    ),
}


# ===========================================================================
# Low level helpers
# ===========================================================================
def _get_device():
    """Return the best torch device, using ComfyUI's manager when available."""
    if _HAS_COMFY:
        try:
            return model_management.get_torch_device()
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_name_choices():
    names = [_NONE]
    for name in UPSCALE_MODEL_CATALOG:
        names.append(name)
    if folder_paths is not None:
        for name in folder_paths.get_filename_list("upscale_models"):
            if name not in names:
                names.append(name)
    return names


def _safe_model_filename(name):
    if not name or name == _NONE:
        return None
    filename = os.path.basename(name)
    if filename != name or filename in {".", ".."}:
        raise ValueError(
            "SmartEnsembleUpscale: invalid upscale model name {!r}.".format(name)
        )
    return filename


def _download_upscale_model(filename, url):
    dest_dir = folder_paths.get_folder_paths("upscale_models")[0]
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.abspath(os.path.join(dest_dir, filename))
    if not folder_paths.is_within_directory(dest_dir, dest):
        raise ValueError(
            "SmartEnsembleUpscale: refused to write {!r} outside upscale_models.".format(
                filename
            )
        )

    logging.info("SmartEnsembleUpscale: downloading %s", filename)
    tmp = dest + ".download"
    req = urllib.request.Request(
        url, headers={"User-Agent": "ComfyUI-SmartEnsembleUpscale"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as src, open(tmp, "wb") as out:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    logging.info("SmartEnsembleUpscale: saved %s", dest)
    return dest


def _load_named_upscale_model(name, download_missing):
    filename = _safe_model_filename(name)
    if filename is None:
        return None
    if folder_paths is None or ModelLoader is None:
        raise RuntimeError(
            "SmartEnsembleUpscale: ComfyUI model loading is not available."
        )

    path = folder_paths.get_full_path("upscale_models", filename)
    if path is None:
        url = UPSCALE_MODEL_CATALOG.get(filename)
        if not download_missing:
            raise FileNotFoundError(
                "SmartEnsembleUpscale: {!r} is missing. Enable download_missing "
                "or place the file in models/upscale_models/.".format(filename)
            )
        if url is None:
            raise FileNotFoundError(
                "SmartEnsembleUpscale: {!r} is missing and is not in the "
                "download catalog.".format(filename)
            )
        _download_upscale_model(filename, url)
        path = folder_paths.get_full_path_or_raise("upscale_models", filename)

    sd = comfy_utils.load_torch_file(path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy_utils.state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(out, ImageModelDescriptor):
        raise Exception("Upscale model must be a single-image model.")
    return out


def _preset_model_names(preset, model_1_name, model_2_name, model_3_name):
    picked = MODEL_PRESETS.get(preset)
    if picked is None:
        return model_1_name, model_2_name, model_3_name
    return picked


def _collect_models(model_1_name, model_2_name, model_3_name, download_missing):
    models = []
    for name in (model_1_name, model_2_name, model_3_name):
        named = _load_named_upscale_model(name, download_missing)
        if named is not None:
            models.append(named)
    if not models:
        raise ValueError(
            "SmartEnsembleUpscale: choose at least one upscale model."
        )
    return models


def _model_scale(upscale_model, fallback=4):
    """
    Best-effort extraction of a model's integer upscale factor.

    spandrel's ImageModelDescriptor (what modern ComfyUI hands us) exposes a
    ``.scale`` attribute.  We fall back to a couple of other common names and
    finally to ``fallback`` (most ESRGAN models are 4x).
    """
    for attr in ("scale", "scale_factor", "upscale_factor"):
        val = getattr(upscale_model, attr, None)
        if isinstance(val, (int, float)) and val >= 1:
            return int(round(val))
    # Some descriptors nest the real module in ``.model``.
    inner = getattr(upscale_model, "model", None)
    if inner is not None:
        for attr in ("scale", "scale_factor", "upscale_factor"):
            val = getattr(inner, attr, None)
            if isinstance(val, (int, float)) and val >= 1:
                return int(round(val))
    return int(fallback)


def _gaussian_kernel1d(sigma, device, dtype):
    """Return a normalised 1D Gaussian kernel as a tensor."""
    radius = max(1, int(math.ceil(sigma * 3)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur(img_bchw, sigma):
    """
    Separable Gaussian blur for a (B, C, H, W) tensor.

    Uses reflection padding so borders are not darkened.
    """
    if sigma <= 0:
        return img_bchw
    device, dtype = img_bchw.device, img_bchw.dtype
    k1d = _gaussian_kernel1d(sigma, device, dtype)
    radius = (k1d.numel() - 1) // 2
    c = img_bchw.shape[1]

    # Horizontal then vertical pass, done per-channel via grouped conv.
    kx = k1d.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k1d.view(1, 1, -1, 1).repeat(c, 1, 1, 1)

    x = F.pad(img_bchw, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x


def _sobel_edges(img_bchw):
    """
    Return a single-channel (B, 1, H, W) edge-magnitude map in [0, 1].

    Computed on the luminance of the image with the classic Sobel operator.
    """
    device, dtype = img_bchw.device, img_bchw.dtype
    # Luminance (Rec. 601-ish) -> (B,1,H,W)
    if img_bchw.shape[1] >= 3:
        r, g, b = img_bchw[:, 0:1], img_bchw[:, 1:2], img_bchw[:, 2:3]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        lum = img_bchw[:, 0:1]

    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                      device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                      device=device, dtype=dtype).view(1, 1, 3, 3)

    lum_p = F.pad(lum, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(lum_p, kx)
    gy = F.conv2d(lum_p, ky)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)

    # Scale by a high percentile, not the single strongest edge. One bright
    # contour would otherwise push every other edge down to "flat".
    # quantile() rejects more than 2^24 values, so large images are sampled.
    flat = mag.flatten(1)
    step = max(1, (flat.shape[1] + (1 << 20) - 1) // (1 << 20))
    sample = flat[:, ::step].contiguous()
    scale = torch.quantile(sample.float(), 0.90, dim=1).clamp(min=0.02)
    scale = scale.to(dtype=mag.dtype).view(-1, 1, 1, 1)
    return (mag / scale).clamp(0.0, 1.0)


def _reduce_overlay(img_bchw, amount):
    """
    Fade a fine, low-amplitude overlay such as the weave Qwen leaves on
    flat areas. Strong edges keep their detail.
    """
    if amount <= 0:
        return img_bchw
    amount = float(max(0.0, min(1.0, amount)))
    sigma = 0.6 + 0.6 * amount
    low = _gaussian_blur(img_bchw, sigma)
    high = img_bchw - low
    amp = _gaussian_blur(high.abs().mean(dim=1, keepdim=True), 0.6)
    texture = torch.exp(-amp / 0.04)
    return (img_bchw - high * texture * amount).clamp(0.0, 1.0)


def _add_noise(img_bchw, amount, seed):
    """
    Add seeded luminance grain. The same seed repeats the same pattern.
    """
    if amount <= 0:
        return img_bchw
    amount = float(max(0.0, min(1.0, amount)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63))
    grain = torch.randn(
        img_bchw.shape[0], 1, img_bchw.shape[2], img_bchw.shape[3],
        generator=generator, dtype=torch.float32,
    )
    grain = grain.to(device=img_bchw.device, dtype=img_bchw.dtype)
    return (img_bchw + grain * amount).clamp(0.0, 1.0)


def _luma(img_bchw):
    r, g, b = img_bchw[:, 0:1], img_bchw[:, 1:2], img_bchw[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _rgb_vec(img_bchw, values):
    return img_bchw.new_tensor(values).view(1, 3, 1, 1)


def _grade(img_bchw, sat, contrast, lift, gain):
    lum = _luma(img_bchw)
    x = lum + (img_bchw - lum) * sat
    x = (x - 0.5) * contrast + 0.5
    x = x * _rgb_vec(img_bchw, gain) + _rgb_vec(img_bchw, lift)
    return x.clamp(0.0, 1.0)


def _teal_orange(img_bchw):
    lum = _luma(img_bchw)
    sat = lum + (img_bchw - lum) * 1.15
    teal = sat * _rgb_vec(img_bchw, (0.82, 1.02, 1.08))
    orange = sat * _rgb_vec(img_bchw, (1.16, 1.00, 0.82))
    x = teal * (1.0 - lum) + orange * lum
    return ((x - 0.5) * 1.08 + 0.5).clamp(0.0, 1.0)


def _infrared(img_bchw):
    r, g, b = img_bchw[:, 0:1], img_bchw[:, 1:2], img_bchw[:, 2:3]
    lum = (0.15 * r + 0.75 * g + 0.10 * b).clamp(0.0, 1.0)
    return torch.cat((lum * 1.05, lum * 0.92, lum * 0.82), dim=1).clamp(0.0, 1.0)


def _redscale(img_bchw):
    r, g, b = img_bchw[:, 0:1], img_bchw[:, 1:2], img_bchw[:, 2:3]
    out_r = r * 0.75 + g * 0.25
    out_g = r * 0.45 + g * 0.35
    out_b = r * 0.22 + b * 0.12
    return torch.cat((out_r, out_g, out_b), dim=1).clamp(0.0, 1.0)


# sat, contrast, lift RGB, gain RGB. Color only, so detail stays put.
_PHOTO_NONE = "None"
_PHOTO_FILTERS = {
    "Sepia": (0.0, 1.05, (0.02, 0.0, 0.0), (1.18, 0.93, 0.58)),
    "Black and White": (0.0, 1.05, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "Silver Gelatin": (0.0, 1.12, (0.0, 0.0, 0.0), (0.98, 1.0, 1.04)),
    "Noir": (0.0, 1.5, (0.0, 0.0, 0.0), (0.95, 0.95, 0.95)),
    "Warm": (1.05, 1.0, (0.0, 0.0, 0.0), (1.08, 1.02, 0.92)),
    "Cool": (1.0, 1.02, (0.0, 0.0, 0.0), (0.92, 1.0, 1.08)),
    "Golden Hour": (1.12, 1.06, (0.0, 0.0, 0.0), (1.16, 1.04, 0.82)),
    "Sunset": (1.2, 1.12, (0.0, 0.0, 0.0), (1.22, 0.96, 0.78)),
    "Amber": (0.9, 1.02, (0.0, 0.0, 0.0), (1.16, 1.0, 0.72)),
    "Autumn": (1.15, 1.08, (0.0, 0.0, 0.0), (1.14, 1.0, 0.78)),
    "Winter": (0.7, 1.06, (0.02, 0.02, 0.03), (0.9, 0.98, 1.12)),
    "Rose": (0.95, 0.98, (0.0, 0.0, 0.0), (1.12, 0.94, 1.02)),
    "Mint": (0.9, 1.0, (0.0, 0.0, 0.0), (0.9, 1.08, 1.0)),
    "Lavender": (0.85, 0.98, (0.0, 0.0, 0.0), (1.02, 0.94, 1.12)),
    "Chocolate": (0.75, 1.05, (0.0, 0.0, 0.0), (1.1, 0.92, 0.75)),
    "Emerald": (0.9, 1.05, (0.0, 0.02, 0.0), (0.85, 1.1, 0.9)),
    "Cyanotype": (0.0, 1.12, (0.0, 0.0, 0.0), (0.55, 0.78, 1.18)),
    "Teal and Orange": _teal_orange,
    "Bleach Bypass": (0.45, 1.35, (0.0, 0.0, 0.0), (1.02, 1.0, 0.96)),
    "Cross Process": (1.3, 1.18, (0.0, 0.04, 0.02), (1.06, 0.96, 1.14)),
    "Redscale": _redscale,
    "Kodachrome": (1.22, 1.14, (0.0, 0.0, 0.0), (1.1, 0.98, 0.9)),
    "Portra": (0.82, 0.92, (0.035, 0.025, 0.02), (1.04, 1.01, 0.96)),
    "Velvia": (1.45, 1.2, (0.0, 0.0, 0.0), (1.04, 1.02, 0.98)),
    "Polaroid": (0.8, 0.88, (0.05, 0.04, 0.03), (1.06, 1.02, 0.9)),
    "Vintage": (0.65, 0.86, (0.05, 0.03, 0.01), (1.08, 0.98, 0.82)),
    "Faded": (0.75, 0.8, (0.08, 0.07, 0.06), (0.92, 0.94, 0.96)),
    "Lomo": (1.35, 1.28, (0.0, 0.0, 0.0), (1.08, 0.98, 1.05)),
    "Chrome": (1.35, 1.22, (0.0, 0.0, 0.0), (0.98, 1.02, 1.06)),
    "Infrared": _infrared,
    "Cinematic": (0.9, 1.12, (0.01, 0.01, 0.02), (1.06, 0.98, 0.94)),
    "Dramatic": (0.8, 1.32, (0.0, 0.0, 0.0), (0.96, 0.98, 1.04)),
    "High Contrast": (1.1, 1.45, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "Soft": (0.9, 0.82, (0.04, 0.03, 0.03), (1.0, 1.0, 1.0)),
    "Pastel": (0.6, 0.78, (0.06, 0.05, 0.06), (1.02, 1.0, 1.02)),
    "High Key": (0.95, 0.85, (0.1, 0.09, 0.09), (1.0, 1.0, 1.0)),
    "Low Key": (0.85, 1.2, (0.0, 0.0, 0.0), (0.78, 0.78, 0.82)),
    "Night": (0.75, 1.1, (0.0, 0.0, 0.02), (0.72, 0.82, 1.12)),
}


def _apply_photo_filter(img_bchw, name):
    """Color grade. Unknown names and None leave the image unchanged."""
    spec = _PHOTO_FILTERS.get(name)
    if spec is None:
        return img_bchw
    if callable(spec):
        return spec(img_bchw).clamp(0.0, 1.0)
    sat, contrast, lift, gain = spec
    return _grade(img_bchw, sat, contrast, lift, gain)


def _reduce_grid(img_bchw, amount):
    """
    Remove a 2-pixel diamond grid on flat areas before upscaling.

    Only the part that stays in phase is removed. The upscaled image is left
    alone, so the models' detail is not smeared back out.
    """
    if amount <= 0:
        return img_bchw
    amount = float(max(0.0, min(1.0, amount)))
    _, _, h, w = img_bchw.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    if h2 < 4 or w2 < 4:
        return img_bchw
    x = img_bchw[:, :, :h2, :w2]
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    chk = 0.25 * (a - b - c + d)
    chk_l = chk.mean(dim=1, keepdim=True)
    smooth = _gaussian_blur(chk_l, 2.0)
    weight = ((chk_l * smooth) > 0).float()
    weight = weight * (smooth.abs() / (smooth.abs() + 0.01)).clamp(0.0, 1.0)
    weight = weight * amount
    a = a - weight * chk
    b = b + weight * chk
    c = c + weight * chk
    d = d - weight * chk
    out = img_bchw.clone()
    out[:, :, 0:h2:2, 0:w2:2] = a
    out[:, :, 0:h2:2, 1:w2:2] = b
    out[:, :, 1:h2:2, 0:w2:2] = c
    out[:, :, 1:h2:2, 1:w2:2] = d
    return out.clamp(0.0, 1.0)


def _tile_starts(total, tile, step):
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, step))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def _tile_geometry(h, w, tile_size, overlap):
    tile_size = max(16, int(tile_size))
    overlap = int(max(0, min(overlap, tile_size - 8)))
    step = max(1, tile_size - overlap)
    return _tile_starts(h, tile_size, step), _tile_starts(w, tile_size, step)


_BAND_SIGMA = 1.5


def _split_bands(img_bchw, sigma):
    low = _gaussian_blur(img_bchw, sigma)
    return low, img_bchw - low


def _combine_bands(lows, highs, weights):
    """
    Crossfade color, keep one model's detail.

    ``weights`` blend the low bands. The high band is taken from the model
    with the largest weight, so disagreeing detail is not averaged away.
    """
    low = (torch.stack(lows, 0) * weights).sum(dim=0)
    high_stack = torch.stack(highs, 0)
    winner = weights.argmax(dim=0, keepdim=True)
    winner = winner.expand(
        -1, high_stack.shape[1], high_stack.shape[2], -1, -1,
    )
    high = high_stack.gather(0, winner).squeeze(0)
    return low + high


def _blend_weights(details, edge, blend_mode):
    """
    Per-model weights, shape (N, B, 1, H, W), summing to 1 over models.

    content_aware keeps the first model on edges. That model owns the
    detail, so a noisier model cannot win just by having more texture.
    Flat areas take the model with the least local detail.
    """
    if blend_mode == "average":
        return torch.ones_like(details) / details.shape[0]
    level = details / details.amax(dim=0, keepdim=True).clamp(min=1e-6)
    if blend_mode == "sharpest":
        return torch.softmax(level * 4.0, dim=0)
    if blend_mode == "smoothest":
        return torch.softmax(-level * 4.0, dim=0)
    smooth = torch.softmax(-level * 4.0, dim=0)
    primary = torch.zeros_like(smooth)
    primary[0] = 1
    e = edge.unsqueeze(0)
    return e * primary + (1.0 - e) * smooth


def _local_detail(img_bchw, sigma=1.5):
    """
    Per-pixel high-frequency energy map (B, 1, H, W).

    Defined as the local magnitude of (image - blur(image)), averaged over
    channels then slightly blurred so the weight varies smoothly.  A model
    whose output has higher local detail is the "sharper" model.
    """
    high = img_bchw - _gaussian_blur(img_bchw, sigma)
    energy = high.abs().mean(dim=1, keepdim=True)
    energy = _gaussian_blur(energy, sigma)
    return energy


def _make_gaussian_window(h, w, device, dtype):
    """
    A 2D Gaussian weighting window of size (h, w), peak 1.0 at the centre,
    tapering towards the edges.  Used to feather overlapping tiles.
    """
    ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    # sigma chosen so the edge weight is ~0.05 -> smooth but non-zero borders.
    sigma = 0.5
    wy = torch.exp(-(ys ** 2) / (2 * sigma * sigma))
    wx = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    win = torch.outer(wy, wx)  # (h, w)
    win = win.clamp(min=1e-3)
    return win.view(1, 1, h, w)


# ===========================================================================
# Core upscaling routines
# ===========================================================================
def _as_rgb(img_bchw):
    """
    ESRGAN-style models take 3-channel RGB.

    Extra channels (a mask or a leftover alpha) are dropped. Reattaching them
    makes Save/Preview treat the result as RGBA and wash the picture out.
    """
    c = img_bchw.shape[1]
    if c == 3:
        return img_bchw
    if c < 3:
        return img_bchw[:, :1].expand(-1, 3, -1, -1).contiguous()
    return img_bchw[:, :3]


def _run_model(upscale_model, img_bchw):
    """
    Run a single ESRGAN-style model on a whole (already-tiled) chunk.

    The image is expected as (B, C, H, W) in [0, 1].  Returns the upscaled
    tensor on the same device.
    """
    return upscale_model(img_bchw)


def _gaussian_tiled_scale(img_bchw, upscale_fn, scale, tile_size, overlap,
                          pbar=None, band_sigma=_BAND_SIGMA):
    """
    Seam-free tiled upscaling.

    Overlaps crossfade the low frequency so the color has no seam. Detail
    comes from the tile whose centre is nearest, so the two guesses are not
    averaged together.
    """
    b, c, h, w = img_bchw.shape
    device, dtype = img_bchw.device, img_bchw.dtype

    out_h, out_w = h * scale, w * scale
    low_acc = torch.zeros((b, c, out_h, out_w), device=device, dtype=dtype)
    high_acc = torch.zeros((b, c, out_h, out_w), device=device, dtype=dtype)
    weight = torch.zeros((b, 1, out_h, out_w), device=device, dtype=dtype)
    best = torch.zeros((b, 1, out_h, out_w), device=device, dtype=dtype)
    out_sigma = band_sigma * scale

    ys, xs = _tile_geometry(h, w, tile_size, overlap)
    tile_size = max(16, int(tile_size))

    for y in ys:
        for x in xs:
            if pbar is not None:
                pbar.update(1)
            th = min(tile_size, h - y)
            tw = min(tile_size, w - x)
            tile_in = img_bchw[:, :, y:y + th, x:x + tw]

            tile_out = upscale_fn(tile_in)
            exp_h, exp_w = th * scale, tw * scale
            if tile_out.shape[-2:] != (exp_h, exp_w):
                tile_out = F.interpolate(
                    tile_out, size=(exp_h, exp_w),
                    mode="bicubic", align_corners=False,
                )

            tile_low, tile_high = _split_bands(tile_out, out_sigma)
            win = _make_gaussian_window(exp_h, exp_w, device, dtype)
            oy, ox = y * scale, x * scale
            region = (slice(None), slice(None), slice(oy, oy + exp_h), slice(ox, ox + exp_w))
            low_acc[region] = low_acc[region] + tile_low * win
            covered = best[:, :, oy:oy + exp_h, ox:ox + exp_w]
            take = win > covered
            high_acc[region] = torch.where(take, tile_high, high_acc[region])
            best[:, :, oy:oy + exp_h, ox:ox + exp_w] = torch.where(take, win, covered)
            weight[:, :, oy:oy + exp_h, ox:ox + exp_w] += win

    output = low_acc / weight.clamp(min=1e-6) + high_acc
    return output.clamp(0.0, 1.0)


def _ensemble_upscale(models, img_bchw, target_hw, tile_size, overlap,
                      blend_mode, model_names=None, pbar=None,
                      band_sigma=_BAND_SIGMA):
    """
    Upscale ``img_bchw`` with each model, resize every result to ``target_hw``
    and blend them together content-aware.

    Returns a (B, C, target_h, target_w) tensor in [0, 1].
    """
    device = img_bchw.device
    results = []
    names = list(model_names or [])
    for index, m in enumerate(models):
        scale = _model_scale(m)
        label = names[index] if index < len(names) else "model {}".format(index + 1)
        if pbar is not None:
            pbar.set_description("SmartEnsembleUpscale " + label, refresh=False)
        up = _gaussian_tiled_scale(
            img_bchw,
            lambda t, _m=m: _run_model(_m, t),
            scale, tile_size, overlap, pbar, band_sigma,
        )
        if up.shape[-2:] != tuple(target_hw):
            up = F.interpolate(up, size=tuple(target_hw),
                               mode="bicubic", align_corners=False).clamp(0, 1)
        results.append(up)

    if len(results) == 1:
        return results[0]

    # ---- Content-aware ensemble blending ---------------------------------
    # Reference edge map computed from the average of the model outputs, so
    # it reflects the actual upscaled structure.
    ref = torch.stack(results, dim=0).mean(dim=0)
    edge = _sobel_edges(ref)                       # (B,1,H,W) in [0,1]
    details = torch.stack([_local_detail(r) for r in results], dim=0)  # (N,B,1,H,W)
    weights = _blend_weights(details, edge, blend_mode)
    weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-6)

    if blend_mode == "average":
        stacked = torch.stack(results, dim=0)
        return stacked.mean(dim=0).clamp(0.0, 1.0)

    out_sigma = band_sigma * (target_hw[0] / float(img_bchw.shape[-2]))
    lows = []
    highs = []
    for result in results:
        low, high = _split_bands(result, out_sigma)
        lows.append(low)
        highs.append(high)
    return _combine_bands(lows, highs, weights).clamp(0.0, 1.0)


# ===========================================================================
# The ComfyUI node
# ===========================================================================
class SmartEnsembleUpscale:
    """
    Smart Ensemble Upscale 🔬 - ensemble + frequency-separated + seam-free
    tiled upscaling in a single node.
    """

    DESCRIPTION = (
        "Upscales an image with 1–3 ESRGAN models. Edges keep the first "
        "model's detail and flat areas use the smoothest model. Color is "
        "crossfaded; detail is taken from one model so it is not averaged "
        "away. Frequency split keeps the original colors. Tile overlaps do "
        "the same: color blends, detail comes from the nearer tile."
    )

    @classmethod
    def INPUT_TYPES(cls):
        model_names = _model_name_choices()
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to upscale.",
                }),
                "model_1_name": (model_names, {
                    "default": "4xNomos8kSC.pth",
                    "tooltip": "First model. Its scale sets the enlargement, usually 4×. Default is a natural photo model.",
                }),
                "model_2_name": (model_names, {
                    "default": "RealESRGAN_x4plus.pth",
                    "tooltip": "Second model. none uses only model 1. Default adds general real-world sharpness.",
                }),
                "model_3_name": (model_names, {
                    "default": "4x_NMKD-Superscale-SP_178000_G.pth",
                    "tooltip": "Third model. none skips it. Default is softer on flat areas such as skin.",
                }),
                "download_missing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Download a catalog model if the file is missing. Custom models must already be in models/upscale_models.",
                }),
                "tile_size": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 32,
                    "tooltip": "Size of each piece the model runs. Smaller uses less VRAM.",
                }),
                "tile_overlap": ("INT", {
                    "default": 64, "min": 0, "max": 1024, "step": 8,
                    "tooltip": "How many pixels the pieces overlap. Higher hides seams but takes longer.",
                }),
                "blend_mode": ([
                    "content_aware", "average", "sharpest", "smoothest",
                ], {
                    "default": "content_aware",
                    "tooltip": "content_aware: first model on edges, smoothest model on flat areas. average: equal weight. sharpest: always the sharpest model. smoothest: always the smoothest model.",
                }),
                "frequency_split": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: colors come from the original, detail from the models. Off: the model output is used as-is.",
                }),
                "output_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05,
                    "tooltip": "Scale the result down afterwards. 0.5 on a 4× model gives 2×. 1.0 keeps the full size.",
                }),
                # Keep preset last so older workflows do not shift saved widget values.
                "preset": ([_CUSTOM, *MODEL_PRESETS.keys()], {
                    "default": "Realistic",
                    "tooltip": "Switches the three models. Custom uses the model choices above. Editing a model sets this to Custom.",
                }),
                "texture_smooth": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fades random fine grain before upscaling. 0 is off. Does not remove a dotted grid.",
                }),
                "reduce_grid": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Removes a dotted or diamond grid, such as the Qwen 2.1 VAE pattern, before upscaling. 0 is off.",
                }),
                "noise": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Adds random grain after upscaling. 0 is off. 0.03 is a light grain. The pattern follows noise seed.",
                }),
                "photo_filter": ([_PHOTO_NONE, *_PHOTO_FILTERS.keys()], {
                    "default": _PHOTO_NONE,
                    "tooltip": "Photographic color grade applied after upscaling. None leaves the colors unchanged.",
                }),
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1,
                    "control_after_generate": True,
                    "tooltip": "Seed for the added grain. The same seed repeats the same noise.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("The upscaled image.",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    # -------------------------------------------------------------------
    def upscale(self, image, tile_size, tile_overlap, blend_mode,
                frequency_split, output_scale, preset="Realistic",
                model_1_name=_NONE, model_2_name=_NONE, model_3_name=_NONE,
                download_missing=True, texture_smooth=0.0, reduce_grid=0.0,
                noise=0.03, photo_filter=_PHOTO_NONE, noise_seed=0):
        """
        Main entry point called by ComfyUI.

        image : IMAGE tensor (B, H, W, C) float32 in [0, 1]
        """
        device = _get_device()

        model_1_name, model_2_name, model_3_name = _preset_model_names(
            preset, model_1_name, model_2_name, model_3_name,
        )
        model_names = [
            name for name in (model_1_name, model_2_name, model_3_name)
            if name and name != _NONE
        ]
        models = _collect_models(
            model_1_name, model_2_name, model_3_name, download_missing,
        )

        # Move models to the compute device (and back to CPU afterwards to
        # keep VRAM free).
        moved = []
        for m in models:
            try:
                m.to(device)
            except Exception:
                pass
            moved.append(m)

        pbar = None
        try:
            # ComfyUI IMAGE is (B, H, W, C); convert to (B, C, H, W).
            img = image.movedim(-1, -3).to(device).float().clamp(0.0, 1.0)
            rgb = _as_rgb(img)
            if reduce_grid > 0:
                rgb = _reduce_grid(rgb, reduce_grid)
            if texture_smooth > 0:
                rgb = _reduce_overlay(rgb, texture_smooth)
            _, _, in_h, in_w = rgb.shape

            # Common target size dictated by the primary model's scale factor.
            primary_scale = _model_scale(models[0])
            target_h, target_w = in_h * primary_scale, in_w * primary_scale
            target_hw = (target_h, target_w)
            ys, xs = _tile_geometry(in_h, in_w, tile_size, tile_overlap)
            n_tiles = len(ys) * len(xs)
            pbar = tqdm(
                total=len(moved) * n_tiles,
                desc="SmartEnsembleUpscale",
                unit="tile",
            )

            if frequency_split:
                result = self._frequency_separated_upscale(
                    models, rgb, target_hw, tile_size, tile_overlap,
                    blend_mode, model_names, pbar,
                )
            else:
                result = _ensemble_upscale(
                    models, rgb, target_hw, tile_size, tile_overlap,
                    blend_mode, model_names, pbar,
                )
            # Optional down-scale of the final result.  e.g. run a 4x model
            # but output at 2x (output_scale=0.5) for higher effective
            # quality than a direct 2x model.
            if output_scale < 0.999:
                final_h = max(1, int(round(target_h * output_scale)))
                final_w = max(1, int(round(target_w * output_scale)))
                result = F.interpolate(
                    result, size=(final_h, final_w),
                    mode="bicubic", align_corners=False,
                ).clamp(0.0, 1.0)

            result = _apply_photo_filter(result, photo_filter)
            if noise > 0:
                result = _add_noise(result, noise, noise_seed)

            # Back to ComfyUI IMAGE format (B, H, W, C) on CPU.
            out = result.movedim(-3, -1).to("cpu").float().clamp(0.0, 1.0)
            return (out,)

        finally:
            if pbar is not None:
                pbar.close()
            # Free VRAM: push models back to CPU.
            for m in moved:
                try:
                    m.to("cpu")
                except Exception:
                    pass
            if _HAS_COMFY:
                try:
                    model_management.soft_empty_cache()
                except Exception:
                    pass
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    def _frequency_separated_upscale(self, models, img, target_hw, tile_size,
                                     tile_overlap, blend_mode, model_names=None,
                                     pbar=None):
        """
        Frequency-separated upscaling.

        * Low frequency (colour / tone): a bicubic enlarge of the original,
          then the same blur that is removed from the model output. The two
          bands meet, so detail is not doubled or left missing at the cutoff.
        * High frequency (detail): what the ensemble added above that blur.

        Recombining keeps the original color and the model's detail.
        """
        target_h, target_w = target_hw
        blur_sigma = _BAND_SIGMA
        in_h = img.shape[-2]
        model_sigma = blur_sigma * (target_h / float(in_h))

        model_up = _ensemble_upscale(
            models, img, target_hw, tile_size, tile_overlap, blend_mode,
            model_names, pbar, _BAND_SIGMA,
        )
        base = F.interpolate(
            img, size=(target_h, target_w),
            mode="bicubic", align_corners=False,
        )
        low_up, _ = _split_bands(base, model_sigma)
        _, high = _split_bands(model_up, model_sigma)
        return (low_up + high).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# ComfyUI registration tables
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SmartEnsembleUpscale": SmartEnsembleUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartEnsembleUpscale": "Smart Ensemble Upscale 🔬",
}
