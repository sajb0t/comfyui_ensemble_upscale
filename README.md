# Smart Ensemble Upscale

A ComfyUI node that upscales one image with one to three ESRGAN models and puts the result back together so color and detail do not cancel each other out.

Find it under **image/upscaling** as **Smart Ensemble Upscale**.

## What it does

The models are chosen from dropdowns. The lists show every file already in `models/upscale_models`, including your own. A preset fills the three dropdowns. **Realistic** is the default: 4xNomos8kSC, RealESRGAN_x4plus, and 4x_NMKD-Superscale. The first model sets the scale, usually 4×. One model skips the blend.

The image is split into tiles so it fits in VRAM. Tile overlaps crossfade color and tone only. Detail comes from the tile whose center is nearest, so two different guesses are not averaged into a soft seam.

With more than one model, edges keep detail from the first model. Flat areas take detail from the smoothest model. Color is crossfaded between them. Other blend modes are average, sharpest, and smoothest.

`frequency_split` is on by default. The original is enlarged with bicubic and blurred with the same cutoff that is removed from the model result. Color and tone stay from the original. Detail comes from the models.

Final size is `original × model scale × output_scale`. The default `output_scale` of **0.5** turns a 4× model into a 2× image. **1.0** keeps the full enlargement.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sajb0t/comfyui_ensamble_upscale.git
```

Restart ComfyUI. Connect an image to `image`, then send the output to Save Image or Preview Image. No extra Python packages are required beyond ComfyUI.

`download_missing` downloads a known catalog model into `models/upscale_models` when that file is not already there. Your own files are never downloaded. Turn the switch off to load only what is already on disk.

Catalog models:

- 4x-UltraSharp.pth
- 4x-AnimeSharp.pth
- RealESRGAN_x4plus.pth
- RealESRGAN_x4plus_anime_6B.pth
- 4x_NMKD-Siax_200k.pth
- 4x_foolhardy_Remacri.pth
- 4x_NMKD-Superscale-SP_178000_G.pth
- 4xNomos8kSC.pth
- RealESRGAN_x2plus.pth

Any single-image upscale model that ComfyUI can load (`.pth` or `.safetensors`) can be used if you place it in `models/upscale_models`.

## Presets

| Preset | Models |
| --- | --- |
| Realistic | 4xNomos8kSC, RealESRGAN_x4plus, 4x_NMKD-Superscale |
| Anime | 4x-AnimeSharp, RealESRGAN_x4plus_anime_6B, 4x-UltraSharp |
| Sharp | 4x-UltraSharp, 4x_foolhardy_Remacri, 4xNomos8kSC |
| Smooth | 4x_NMKD-Superscale, 4xNomos8kSC, RealESRGAN_x4plus |
| 2x | RealESRGAN_x2plus only |
| Custom | Uses the three dropdowns as they are |

Editing a model dropdown sets the preset to Custom.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `model_1_name` | 4xNomos8kSC.pth | First model. Sets the enlargement. |
| `model_2_name` | RealESRGAN_x4plus.pth | Second model. `none` skips it. |
| `model_3_name` | 4x_NMKD-Superscale-SP_178000_G.pth | Third model. `none` skips it. |
| `download_missing` | on | Download a missing catalog model. |
| `tile_size` | 512 | Input tile size. Smaller uses less VRAM. |
| `tile_overlap` | 64 | Overlap between tiles, in input pixels. |
| `blend_mode` | content_aware | How the models are combined. |
| `frequency_split` | on | Keep original color, take detail from the models. |
| `output_scale` | 0.5 | Scale the result down after upscaling. 0.25–1.0. |
| `preset` | Realistic | Switches the three models. |
| `texture_smooth` | 0 | Fade fine grain before upscaling. 0 is off. |
| `reduce_grid` | 0 | Remove a 2-pixel checker from the source. 0 is off. |
| `noise` | 0.03 | Add grain after upscaling. 0 is off. |
| `photo_filter` | None | Color grade after upscaling. |
| `noise_seed` | 0 | Seed for the grain. The same seed repeats the same pattern. |

`texture_smooth` and `reduce_grid` stay off unless you raise them. They are for sources that already have grain or a fine checker, not for every photo.

## Photo filters

`photo_filter` changes color and contrast only. **None** leaves the picture alone. The list includes Sepia, Black and White, Silver Gelatin, Noir, Warm, Cool, Golden Hour, Sunset, Amber, Autumn, Winter, Rose, Mint, Lavender, Chocolate, Emerald, Cyanotype, Teal and Orange, Bleach Bypass, Cross Process, Redscale, Kodachrome, Portra, Velvia, Polaroid, Vintage, Faded, Lomo, Chrome, Infrared, Cinematic, Dramatic, High Contrast, Soft, Pastel, High Key, Low Key, and Night.

## Notes

The node takes a ComfyUI image, a float tensor shaped `(batch, height, width, channels)` in the range 0–1. Extra channels are dropped. The output is always RGB.

Models are moved to the GPU for the run and back to the CPU afterwards.
