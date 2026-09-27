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

try:
    import cv2
except ImportError:
    cv2 = None

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
    "4xFaceUpDAT.pth":
        "https://github.com/Phhofm/models/releases/download/4xFaceUpDAT_Series/4xFaceUpDAT.pth",
}
_FACE_MODEL = "4xFaceUpDAT.pth"
_NONE = "none"
_CUSTOM = "Custom"


def _preset_look(noise):
    """Shared look. Only the grain changes between image types."""
    return {
        "noise": noise,
        "blend_mode": "content_aware",
        "frequency_split": True,
        "texture_smooth": 0.0,
        "reduce_grid": 0.0,
        "photo_filter": "None",
    }


# None means the widgets are used as they are.
MODEL_PRESETS = {
    "Realistic": {
        "models": (
            "4xNomos8kSC.pth",
            "RealESRGAN_x4plus.pth",
            "4x_NMKD-Superscale-SP_178000_G.pth",
        ),
        "settings": _preset_look(0.03),
    },
    "Anime": {
        "models": (
            "4x-AnimeSharp.pth",
            "RealESRGAN_x4plus_anime_6B.pth",
            "4x-UltraSharp.pth",
        ),
        "settings": _preset_look(0.0),
    },
    "Cartoon": {
        "models": (
            "4x-UltraSharp.pth",
            "4x_foolhardy_Remacri.pth",
            "4x_NMKD-Siax_200k.pth",
        ),
        "settings": _preset_look(0.0),
    },
    "Sharp": {
        "models": (
            "4x-UltraSharp.pth",
            "4x_foolhardy_Remacri.pth",
            "4xNomos8kSC.pth",
        ),
        "settings": _preset_look(0.02),
    },
    "Smooth": {
        "models": (
            "4x_NMKD-Superscale-SP_178000_G.pth",
            "4xNomos8kSC.pth",
            "RealESRGAN_x4plus.pth",
        ),
        "settings": _preset_look(0.04),
    },
    "2x": {
        "models": (
            "RealESRGAN_x2plus.pth",
            _NONE,
            _NONE,
        ),
        "settings": _preset_look(0.03),
    },
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
    return picked["models"]


def _preset_settings(preset):
    picked = MODEL_PRESETS.get(preset)
    if picked is None:
        return None
    return picked["settings"]


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


def _reflect_pad(img, left, right, top, bottom):
    """Reflect pad. A pad wider than the image is applied in legal steps."""
    x = img
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    while left or right or top or bottom:
        _, _, h, w = x.shape
        if w <= 1 and (left or right):
            x = F.pad(x, (left, right, 0, 0), mode="replicate")
            left = right = 0
            continue
        if h <= 1 and (top or bottom):
            x = F.pad(x, (0, 0, top, bottom), mode="replicate")
            top = bottom = 0
            continue
        pl = min(left, w - 1)
        pr = min(right, w - 1)
        pt = min(top, h - 1)
        pb = min(bottom, h - 1)
        x = F.pad(x, (pl, pr, pt, pb), mode="reflect")
        left -= pl
        right -= pr
        top -= pt
        bottom -= pb
    return x


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

    x = _reflect_pad(img_bchw, radius, radius, 0, 0)
    x = F.conv2d(x, kx, groups=c)
    x = _reflect_pad(x, 0, 0, radius, radius)
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
# Weight gap below this crossfades the two leading detail sources.
# A wider gap keeps a hard pick so disagreeing detail is not averaged away.
_HANDOFF = 0.2


def _split_bands(img_bchw, sigma):
    low = _gaussian_blur(img_bchw, sigma)
    return low, img_bchw - low


def _free_vram():
    if _HAS_COMFY:
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vram_tight(device, need_bytes):
    """True when the next CUDA allocation of about ``need_bytes`` may fail."""
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return False
    free, _total = torch.cuda.mem_get_info(device)
    return int(free) < int(need_bytes) + (64 << 20)


def _model_to(model, device):
    try:
        model.to(device)
    except Exception:
        pass


def _row_strip(height, width, channels):
    row_bytes = max(1, int(width) * int(channels) * 4)
    return max(8, min(int(height), max(1, (32 << 20) // row_bytes)))


def _reflect_index(index, length):
    length = int(length)
    if length <= 1:
        return torch.zeros_like(index)
    period = 2 * length - 2
    index = torch.remainder(index, period)
    return torch.where(index < length, index, period - index)


def _on_device(tensor, device):
    if tensor.device == device:
        return tensor
    return tensor.to(device)


def _rows_with_context(img, y0, y1, radius):
    """Rows ``y0:y1`` plus ``radius`` of real or reflected context on each side."""
    _, _, h, _ = img.shape
    radius = int(radius)
    if radius <= 0:
        return img[:, :, y0:y1, :]
    top_have = min(radius, max(0, y0))
    bot_have = min(radius, max(0, h - y1))
    sl = img[:, :, max(0, y0 - top_have):min(h, y1 + bot_have), :]
    pad_top = radius - top_have
    pad_bot = radius - bot_have
    if pad_top or pad_bot:
        sl = _reflect_pad(sl, 0, 0, pad_top, pad_bot)
    return sl


def _mix_high(pieces, weights):
    """Detail from the leading piece. The runner-up mixes in only near a tie."""
    if len(pieces) < 2:
        return pieces[0]
    vals, idx = weights.topk(2, dim=0)
    gap = vals[0] - vals[1]
    mix = (1.0 - gap / _HANDOFF).clamp(0.0, 1.0) * 0.5
    winner = torch.zeros_like(pieces[0])
    runner = torch.zeros_like(pieces[0])
    for i, piece in enumerate(pieces):
        winner = torch.where(idx[0] == i, piece, winner)
        runner = torch.where(idx[1] == i, piece, runner)
    return winner * (1.0 - mix) + runner * mix


def _combine_bands(lows, highs, weights):
    """
    Crossfade color and hand off detail, in row strips.

    The full images are never stacked, so the blend does not need another
    copy of every model result.
    """
    ref = lows[0]
    b, c, h, w = ref.shape
    work = weights.device
    out_device = work
    if _vram_tight(work, ref.numel() * ref.element_size()):
        out_device = torch.device("cpu")
    out = torch.empty((b, c, h, w), device=out_device, dtype=ref.dtype)
    step = _row_strip(h, w, c)
    n = len(lows)
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        wstrip = _on_device(weights[:, :, :, y0:y1, :], work)
        low = _on_device(lows[0][:, :, y0:y1, :], work) * wstrip[0]
        for i in range(1, n):
            low = low + _on_device(lows[i][:, :, y0:y1, :], work) * wstrip[i]
        pieces = [_on_device(high[:, :, y0:y1, :], work) for high in highs]
        mixed = (low + _mix_high(pieces, wstrip)).clamp(0.0, 1.0)
        out[:, :, y0:y1, :] = mixed if mixed.device == out_device else mixed.to(out_device)
    return out


def _plain_enlarge(src, out_h, out_w):
    """Bicubic enlarge. This is the picture with no model detail."""
    if src.shape[-2] == out_h and src.shape[-1] == out_w:
        return src
    return F.interpolate(
        src, size=(int(out_h), int(out_w)), mode="bicubic", align_corners=False,
    ).clamp(0.0, 1.0)


def _apply_upscale_strength(result, src, strength):
    """Fade the ensemble toward a plain enlarge, or push its detail further."""
    if strength == 1:
        return result
    plain = _plain_enlarge(src, result.shape[-2], result.shape[-1])
    if plain.device != result.device or plain.dtype != result.dtype:
        plain = plain.to(device=result.device, dtype=result.dtype)
    return (plain + (result - plain) * strength).clamp(0.0, 1.0)


def _area_shrink(img, out_h, out_w):
    """Mean of the source pixels that cover each output pixel."""
    _, _, h, w = img.shape
    out_h = int(out_h)
    out_w = int(out_w)
    if out_h == h and out_w == w:
        return img
    if h % out_h == 0 and w % out_w == 0:
        fh = h // out_h
        fw = w // out_w
        b, c = img.shape[:2]
        pooled = img.reshape(b, c, out_h, fh, out_w, fw).mean(dim=(3, 5))
        return pooled.clamp(0.0, 1.0)
    return F.interpolate(img, size=(out_h, out_w), mode="area").clamp(0.0, 1.0)


def _bicubic_rows(src, row0, row1, out_h, out_w):
    """
    Bicubic sample of output rows ``row0:row1``, matching
    ``interpolate(..., align_corners=False)``. Rows outside the image reflect.
    """
    b = src.shape[0]
    n = int(row1 - row0)
    rows = _reflect_index(torch.arange(row0, row1, device=src.device), out_h)
    cols = torch.arange(out_w, device=src.device)
    rows = rows.to(dtype=src.dtype)
    cols = cols.to(dtype=src.dtype)
    gy = (rows + 0.5) * (2.0 / float(out_h)) - 1.0
    gx = (cols + 0.5) * (2.0 / float(out_w)) - 1.0
    grid = torch.stack((
        gx.view(1, 1, -1).expand(b, n, -1),
        gy.view(1, -1, 1).expand(b, -1, out_w),
    ), dim=-1)
    return F.grid_sample(
        src, grid, mode="bicubic", padding_mode="border", align_corners=False,
    )


def _frequency_combine(src, model_up, sigma):
    """
    ``blur(bicubic(src)) + (model_up - blur(model_up))`` in row strips.

    The enlarged original is never stored as a second full image.
    """
    b, c, target_h, target_w = model_up.shape
    radius = max(1, int(math.ceil(float(sigma) * 3.0)))
    work = src.device
    out_device = model_up.device
    if _vram_tight(work, model_up.numel() * model_up.element_size()):
        out_device = torch.device("cpu")
    out = torch.empty((b, c, target_h, target_w), device=out_device, dtype=model_up.dtype)
    step = _row_strip(target_h, target_w, c)
    for y0 in range(0, target_h, step):
        y1 = min(target_h, y0 + step)
        base_ctx = _bicubic_rows(src, y0 - radius, y1 + radius, target_h, target_w)
        low_base = _gaussian_blur(base_ctx, sigma)
        low_base = low_base[:, :, radius:radius + (y1 - y0), :]
        model_ctx = _on_device(_rows_with_context(model_up, y0, y1, radius), work)
        low_model = _gaussian_blur(model_ctx, sigma)
        model_rows = _on_device(model_up[:, :, y0:y1, :], work)
        high = model_rows - low_model[:, :, radius:radius + (y1 - y0), :]
        piece = (low_base + high).clamp(0.0, 1.0)
        out[:, :, y0:y1, :] = piece if piece.device == out_device else piece.to(out_device)
    return out


def _tile_high_alpha(win, covered):
    """
    1 takes the new tile's detail, 0 keeps the stored tile.

    Pixels with no previous tile are always 1. A near tie of the two
    window weights returns 0.5. A clear lead is a hard pick.
    """
    fresh = covered <= 1e-3
    peak = torch.maximum(win, covered).clamp(min=1e-3)
    gap = (win - covered).abs()
    close = (1.0 - gap / (peak * _HANDOFF)).clamp(0.0, 1.0)
    prefer_new = win >= covered
    alpha = torch.where(prefer_new, 1.0 - 0.5 * close, 0.5 * close)
    return torch.where(fresh, torch.ones_like(alpha), alpha)


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
    comes from the nearer tile centre. The two details blend only where
    their window weights nearly tie.
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
            alpha = _tile_high_alpha(win, covered)
            high_acc[region] = high_acc[region] * (1.0 - alpha) + tile_high * alpha
            best[:, :, oy:oy + exp_h, ox:ox + exp_w] = torch.maximum(covered, win)
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
    names = list(model_names or [])
    average = blend_mode == "average"
    acc = None
    count = 0
    lows = []
    highs = []
    details = []
    edge = None
    out_sigma = band_sigma * (target_hw[0] / float(img_bchw.shape[-2]))
    image_bytes = None

    for index, m in enumerate(models):
        scale = _model_scale(m)
        label = names[index] if index < len(names) else "model {}".format(index + 1)
        if pbar is not None:
            pbar.set_description("SmartEnsembleUpscale " + label, refresh=False)
        # One model on the GPU. The previous full result is not kept beside it.
        _model_to(m, device)
        try:
            up = _gaussian_tiled_scale(
                img_bchw,
                lambda t, _m=m: _run_model(_m, t),
                scale, tile_size, overlap, pbar, band_sigma,
            )
        finally:
            _model_to(m, "cpu")
            _free_vram()
        if up.shape[-2:] != tuple(target_hw):
            up = F.interpolate(up, size=tuple(target_hw),
                               mode="bicubic", align_corners=False).clamp(0, 1)
        if len(models) == 1:
            return up
        if image_bytes is None:
            image_bytes = up.numel() * up.element_size()
        if average:
            count += 1
            if acc is None:
                acc = up
            elif acc.device.type == "cpu" or _vram_tight(device, image_bytes):
                acc = acc.cpu() + up.cpu()
                del up
            else:
                acc = acc + up
                del up
            continue
        if lows and _vram_tight(device, image_bytes * 4):
            lows = [item.cpu() if item.device.type != "cpu" else item for item in lows]
            highs = [item.cpu() if item.device.type != "cpu" else item for item in highs]
            _free_vram()
        if index == 0:
            edge = _sobel_edges(up)
        details.append(_local_detail(up))
        low, high = _split_bands(up, out_sigma)
        del up
        if _vram_tight(device, image_bytes * 3):
            low = low.cpu()
            high = high.cpu()
            _free_vram()
        lows.append(low)
        highs.append(high)

    if average:
        return (acc / count).clamp(0.0, 1.0)

    # Edges come from the first model. That model owns edge detail, so the
    # map must not be taken from an average that has already cancelled it.
    detail_stack = torch.stack([_on_device(item, device) for item in details], 0)
    del details
    weights = _blend_weights(detail_stack, _on_device(edge, device), blend_mode)
    del detail_stack, edge
    weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
    return _combine_bands(lows, highs, weights)


_FACE_DETECTOR = None


def _face_detector(width, height):
    global _FACE_DETECTOR
    if cv2 is None or not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "SmartEnsembleUpscale: face enhance needs OpenCV with FaceDetectorYN."
        )
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "face_detection_yunet_2023mar.onnx",
    )
    if not os.path.isfile(path):
        raise RuntimeError(
            "SmartEnsembleUpscale: the face detector file is missing."
        )
    size = (int(width), int(height))
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = cv2.FaceDetectorYN.create(path, "", size, 0.6, 0.3, 5000)
    else:
        _FACE_DETECTOR.setInputSize(size)
    return _FACE_DETECTOR


def _detect_face_boxes(img_bchw):
    """Frontal faces as (x, y, w, h) on each batch image, in source pixels."""
    boxes = []
    rgb = (img_bchw.detach().clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().numpy()
    for sample in rgb:
        sample = np.ascontiguousarray(sample.transpose(1, 2, 0))
        height, width = sample.shape[:2]
        longest = max(height, width)
        view = sample
        scale = 1.0
        if longest > 1280:
            scale = 1280.0 / float(longest)
            view = cv2.resize(
                sample,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        bgr = np.ascontiguousarray(view[:, :, ::-1])
        detector = _face_detector(bgr.shape[1], bgr.shape[0])
        _count, found = detector.detect(bgr)
        batch = []
        if found is not None and len(found):
            for row in found:
                x0 = int(round(float(row[0]) / scale))
                y0 = int(round(float(row[1]) / scale))
                x1 = int(round(float(row[0] + row[2]) / scale))
                y1 = int(round(float(row[1] + row[3]) / scale))
                x0 = max(0, min(width - 1, x0))
                y0 = max(0, min(height - 1, y0))
                x1 = max(x0 + 1, min(width, x1))
                y1 = max(y0 + 1, min(height, y1))
                if x1 - x0 < 16 or y1 - y0 < 16:
                    continue
                batch.append((x0, y0, x1 - x0, y1 - y0))
        batch.sort(key=lambda item: item[2] * item[3])
        boxes.append(batch)
    return boxes


def _padded_face_box(box, width, height, pad=0.45):
    x, y, w, h = box
    cx = x + w * 0.5
    cy = y + h * 0.5
    x0 = int(round(cx - w * (0.5 + pad)))
    y0 = int(round(cy - h * (0.5 + pad)))
    x1 = int(round(cx + w * (0.5 + pad)))
    y1 = int(round(cy + h * (0.5 + pad)))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _face_blend_mask(rect_h, rect_w, inner, device, dtype):
    """1 on the detected face, fading to 0 across the padding around it."""
    iy0, iy1, ix0, ix1 = inner
    mask = torch.zeros((1, 1, rect_h, rect_w), device=device, dtype=dtype)
    mask[:, :, iy0:iy1, ix0:ix1] = 1
    span = max(1, min(iy1 - iy0, ix1 - ix0))
    sigma = max(1.0, 0.12 * span)
    mask = _gaussian_blur(mask, sigma)
    core_y = max(1, int(round((iy1 - iy0) * 0.15)))
    core_x = max(1, int(round((ix1 - ix0) * 0.15)))
    mask[:, :, iy0 + core_y:iy1 - core_y, ix0 + core_x:ix1 - core_x] = 1
    return mask


def _fit_face(face, height, width):
    if face.shape[-2:] == (height, width):
        return face
    if face.shape[-2] >= height and face.shape[-1] >= width:
        return _area_shrink(face, height, width)
    return F.interpolate(
        face, size=(height, width), mode="bicubic", align_corners=False,
    ).clamp(0.0, 1.0)


def _enhance_faces(result, src, boxes, face_model, tile_size, overlap, pbar=None, strength=1.0):
    """Upscale each face from the source and feather it onto ``result``."""
    device = src.device
    scale = _model_scale(face_model)
    label = "face"
    if pbar is not None:
        pbar.set_description("SmartEnsembleUpscale " + label, refresh=False)
    _model_to(face_model, device)
    try:
        for index, faces in enumerate(boxes):
            in_h, in_w = src.shape[-2:]
            out_h, out_w = result.shape[-2:]
            for box in faces:
                padded = _padded_face_box(box, in_w, in_h)
                if padded is None:
                    continue
                x0, y0, x1, y1 = padded
                crop = src[index:index + 1, :, y0:y1, x0:x1]
                crop_h, crop_w = crop.shape[-2:]
                padded_crop = crop
                extra_h = (16 - crop_h % 16) % 16
                extra_w = (16 - crop_w % 16) % 16
                if extra_h or extra_w:
                    padded_crop = _reflect_pad(crop, 0, extra_w, 0, extra_h)
                if pbar is not None:
                    ys, xs = _tile_geometry(padded_crop.shape[-2], padded_crop.shape[-1], tile_size, overlap)
                    pbar.total += len(ys) * len(xs)
                    pbar.refresh()
                face_up = _gaussian_tiled_scale(
                    padded_crop,
                    lambda tile, _m=face_model: _run_model(_m, tile),
                    scale, tile_size, overlap, pbar, _BAND_SIGMA,
                )
                keep_h = min(face_up.shape[-2], crop_h * scale)
                keep_w = min(face_up.shape[-1], crop_w * scale)
                face_up = face_up[:, :, :keep_h, :keep_w]
                combined = _frequency_combine(crop, face_up, _BAND_SIGMA * scale)
                oy0 = int(round(y0 * out_h / float(in_h)))
                oy1 = int(round(y1 * out_h / float(in_h)))
                ox0 = int(round(x0 * out_w / float(in_w)))
                ox1 = int(round(x1 * out_w / float(in_w)))
                oy0 = max(0, min(out_h - 1, oy0))
                ox0 = max(0, min(out_w - 1, ox0))
                oy1 = max(oy0 + 1, min(out_h, oy1))
                ox1 = max(ox0 + 1, min(out_w, ox1))
                rect_h, rect_w = oy1 - oy0, ox1 - ox0
                fitted = _fit_face(combined, rect_h, rect_w)
                if fitted.device != result.device:
                    fitted = fitted.to(result.device)
                fx0 = int(round((box[0] - x0) * rect_w / float(x1 - x0)))
                fy0 = int(round((box[1] - y0) * rect_h / float(y1 - y0)))
                fx1 = int(round((box[0] + box[2] - x0) * rect_w / float(x1 - x0)))
                fy1 = int(round((box[1] + box[3] - y0) * rect_h / float(y1 - y0)))
                fx0 = max(0, min(rect_w - 1, fx0))
                fy0 = max(0, min(rect_h - 1, fy0))
                fx1 = max(fx0 + 1, min(rect_w, fx1))
                fy1 = max(fy0 + 1, min(rect_h, fy1))
                mask = _face_blend_mask(
                    rect_h, rect_w, (fy0, fy1, fx0, fx1), result.device, result.dtype,
                )
                region = result[index:index + 1, :, oy0:oy1, ox0:ox1]
                result[index:index + 1, :, oy0:oy1, ox0:ox1] = (
                    region + (fitted - region) * mask * strength
                ).clamp(0.0, 1.0)
    finally:
        _model_to(face_model, "cpu")
        _free_vram()
    return result


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
        "crossfaded. Detail comes from one model, and the two leading "
        "details blend only where they nearly tie. Frequency split keeps "
        "the original colors. Tile overlaps do the same. Shrinking the "
        "result averages source pixels."
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
                    "tooltip": "Scale the result down afterwards by averaging source pixels. 0.5 on a 4× model gives 2×. 1.0 keeps the full size.",
                }),
                # Keep preset last so older workflows do not shift saved widget values.
                "preset": ([_CUSTOM, *MODEL_PRESETS.keys()], {
                    "default": "Realistic",
                    "tooltip": "Sets the models and the look for that kind of picture: noise, blend, frequency split, texture smooth, reduce grid, and photo filter. Custom keeps the widgets as they are. Editing one of those sets this to Custom. Tile size, overlap, output scale, and noise seed stay as you set them.",
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
                "face_enhance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "After the ensemble, find each frontal face and sharpen it with the face model. The result is blended back with a soft edge. Off leaves the picture unchanged. No face found leaves it unchanged. Presets do not change this.",
                }),
                "face_model_name": (model_names, {
                    "default": _FACE_MODEL,
                    "tooltip": "Upscale model used only on detected faces. 4xFaceUpDAT is trained on faces. It is downloaded if missing and download missing is on.",
                }),
                "face_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "How strongly the face model replaces the ensemble on each detected face. 1 is the face model as it is. Lower fades it back toward the ensemble. Higher exaggerates that detail. 0 skips the face pass. Presets do not change this.",
                }),
                "upscale_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "How strongly the upscale models replace a plain enlargement. 1 is the ensemble as it is. Lower fades detail back toward that plain resize. Higher exaggerates it. 0 skips the models and only enlarges the picture. Grain, the photo filter, and face strength stay separate. Presets do not change this.",
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
                noise=0.03, photo_filter=_PHOTO_NONE, noise_seed=0,
                face_enhance=False, face_model_name=_FACE_MODEL,
                face_strength=1.0, upscale_strength=1.0):
        """
        Main entry point called by ComfyUI.

        image : IMAGE tensor (B, H, W, C) float32 in [0, 1]
        """
        device = _get_device()

        model_1_name, model_2_name, model_3_name = _preset_model_names(
            preset, model_1_name, model_2_name, model_3_name,
        )
        settings = _preset_settings(preset)
        if settings is not None:
            blend_mode = settings["blend_mode"]
            frequency_split = settings["frequency_split"]
            texture_smooth = settings["texture_smooth"]
            reduce_grid = settings["reduce_grid"]
            noise = settings["noise"]
            photo_filter = settings["photo_filter"]
        model_names = [
            name for name in (model_1_name, model_2_name, model_3_name)
            if name and name != _NONE
        ]
        models = _collect_models(
            model_1_name, model_2_name, model_3_name, download_missing,
        )
        # Each model is moved to the GPU only while its tiles run.
        moved = models

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
            if output_scale < 0.999:
                final_h = max(1, int(round(target_h * output_scale)))
                final_w = max(1, int(round(target_w * output_scale)))
            else:
                final_h, final_w = target_h, target_w

            if upscale_strength == 0:
                result = _plain_enlarge(rgb, final_h, final_w)
            else:
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
                # Shrink by averaging the source pixels that fall into each
                # output pixel. 0.5 on a 4× model is a 2×2 mean, not a second
                # bicubic pass.
                if output_scale < 0.999:
                    result = _area_shrink(result, final_h, final_w)
                result = _apply_upscale_strength(result, rgb, upscale_strength)

            if face_enhance and face_strength > 0:
                face_boxes = _detect_face_boxes(rgb)
                if any(face_boxes) and _safe_model_filename(face_model_name):
                    face_model = _load_named_upscale_model(
                        face_model_name, download_missing,
                    )
                    if face_model is not None:
                        moved.append(face_model)
                        result = _enhance_faces(
                            result, rgb, face_boxes, face_model,
                            tile_size, tile_overlap, pbar, face_strength,
                        )
                elif not any(face_boxes):
                    logging.info("SmartEnsembleUpscale: no frontal face found")

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
        in_h = img.shape[-2]
        model_sigma = _BAND_SIGMA * (target_h / float(in_h))

        model_up = _ensemble_upscale(
            models, img, target_hw, tile_size, tile_overlap, blend_mode,
            model_names, pbar, _BAND_SIGMA,
        )
        return _frequency_combine(img, model_up, model_sigma)


# ---------------------------------------------------------------------------
# ComfyUI registration tables
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SmartEnsembleUpscale": SmartEnsembleUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartEnsembleUpscale": "Smart Ensemble Upscale 🔬",
}
