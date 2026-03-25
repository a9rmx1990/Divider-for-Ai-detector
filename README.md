# Divider Module for AI Image Detection

A high-performance, memory-efficient **image patching engine** designed for AI-generated image detection pipelines.

This module splits any input image into structured, overlapping patches that can be fed into deep learning models (CNNs, Vision Transformers, etc.) to detect artifacts left by AI image generators such as GANs, Stable Diffusion, DALL·E, and Midjourney.

---

## Why This Exists

AI-generated images often contain subtle, localized artifacts — irregular textures, frequency anomalies, or inconsistent lighting — that are invisible at full resolution but become detectable when analyzed patch-by-patch. This Divider module is the **first stage** of that detection pipeline.

## Key Features

| Feature | Description |
|---|---|
| **Streaming Patches** | Uses Python generators — only one patch lives in memory at a time, even on 4K+ images. |
| **No Resizing** | Patches are extracted without any interpolation or scaling, preserving original pixel data. |
| **Overlap Support** | Configurable 0–50% overlap ensures edge artifacts are never missed. |
| **Multi-Scale** | Extract patches at multiple sizes (e.g., 64×64, 128×128, 224×224) in a single pass. |
| **Smart Padding** | Boundary pixels are padded using `reflect`, `edge`, or `constant` modes — no data is lost. |
| **Rich Metadata** | Every patch carries its original (x, y) coordinates, padding info, and scale level. |

## Project Structure

```
├── image_divider.py   # Core divider module (ImageDivider class)
├── test_divider.py    # Test suite + real-image testing CLI
├── .gitignore
└── README.md
```

## Installation

```bash
# Clone the repo
git clone https://github.com/a9rmx1990/Divider-for-Ai-detector.git
cd Divider-for-Ai-detector

# Install dependencies
pip install numpy opencv-python
```

## Quick Start

### Basic Usage (in Python)

```python
from image_divider import ImageDivider

# Create a divider: 224×224 patches with 25% overlap
divider = ImageDivider(patch_sizes=224, overlap_ratio=0.25, padding_mode='reflect')

# Stream patches from any image (JPEG, PNG, WebP)
for patch in divider.stream_patches("photo.jpg"):
    img_data = patch.data              # NumPy array (224, 224, 3)
    coords   = (patch.metadata.x_start, patch.metadata.y_start,
                patch.metadata.x_end,   patch.metadata.y_end)
    
    # Feed into your AI detector model:
    # prediction = model.predict(img_data)
```

### Multi-Scale Extraction

```python
# Extract patches at two scales simultaneously
divider = ImageDivider(patch_sizes=[64, 224], overlap_ratio=0.5)

for patch in divider.stream_patches("suspect_image.png"):
    print(f"Scale: {patch.metadata.patch_size}×{patch.metadata.patch_size}, "
          f"Position: ({patch.metadata.x_start}, {patch.metadata.y_start})")
```

## Testing

```bash
# Run built-in sanity tests (no image needed)
python3 test_divider.py

# Test with a real image (e.g., from Kaggle)
python3 test_divider.py path/to/image.jpg
```

The real-image test will:
1. Print image resolution and file size
2. Run 3 different divider configurations
3. Verify 100% pixel coverage
4. Save sample patches to `divider_output/` for visual inspection

## How It Works

```
┌─────────────────────────────┐
│       Original Image        │
│       (any resolution)      │
└──────────────┬──────────────┘
               │
        ┌──────▼──────┐
        │  Pad edges   │  ← reflect/edge/constant padding
        │  (if needed) │     (no resizing, no distortion)
        └──────┬──────┘
               │
     ┌─────────▼─────────┐
     │  Sliding window    │  ← stride = patch_size × (1 - overlap)
     │  patch extraction  │
     └─────────┬─────────┘
               │
    ┌──────────▼──────────┐
    │  Yield Patch object  │  ← image data + metadata (coords, padding, scale)
    │  (one at a time)     │
    └──────────┬──────────┘
               │
      ┌────────▼────────┐
      │  Downstream AI   │  ← CNN / ViT / frequency analysis / etc.
      │  Detector Model  │
      └─────────────────┘
```

## Patch Metadata

Each patch includes rich metadata for downstream stitching and analysis:

| Field | Type | Description |
|---|---|---|
| `index` | int | Unique patch index |
| `x_start`, `y_start` | int | Top-left corner in the original image |
| `x_end`, `y_end` | int | Bottom-right corner in the original image |
| `patch_size` | int | Patch dimension (e.g., 224) |
| `overlap_ratio` | float | Overlap ratio used |
| `pad_right`, `pad_bottom` | int | Pixels that are padding (not original data) |
| `scale_level` | int | Which multi-scale level this patch belongs to |
| `original_width`, `original_height` | int | Dimensions of the source image |

## Use Cases

This module is designed to support downstream tasks like:

- 🔍 **CNN feature extraction** — localized texture and pattern analysis
- 📊 **Frequency domain analysis** — FFT/DCT on small patches to find GAN fingerprints
- 🎨 **Color histogram analysis** — per-patch brightness and saturation checks
- 🧠 **Vision Transformer inputs** — patches are the native input format for ViTs
- 🔬 **Diffusion artifact detection** — subtle noise patterns in localized regions

## Requirements

- Python 3.8+
- NumPy
- OpenCV (`opencv-python`)

## License

MIT
