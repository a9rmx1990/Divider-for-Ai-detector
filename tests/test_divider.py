"""
Test suite for the ImageDivider module.

Usage:
  # Run all internal tests (no image needed):
    python3 test_divider.py

  # Test with a real image from Kaggle or anywhere else:
    python3 test_divider.py path/to/image.jpg

  The real-image test will:
    1. Print image and patch statistics.
    2. Verify 100% spatial coverage.
    3. Save a few sample patches to an output folder for visual inspection.
"""

import sys
import os
import numpy as np
import cv2
import time

# Ensure we can import from the src/ module easily
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_divider import ImageDivider


# ---------------------------------------------------------------------------
# Internal tests (dummy images)
# ---------------------------------------------------------------------------

def create_dummy_image(w=500, h=400, channels=3):
    """Creates a predictable deterministic colorful image array for testing."""
    img = np.zeros((h, w, channels), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x] = [(x * 2 + y) % 255, (x + y * 2) % 255, (x * y) % 255]
    return img


def test_coverage():
    """Tests that every pixel of the original image is covered by at least one patch."""
    print("=" * 60)
    print("TEST 1: Coverage validation (dummy 502×403 image)")
    print("=" * 60)
    w, h = 502, 403  # Awkward dimensions intentionally
    img = create_dummy_image(w, h)

    divider = ImageDivider(patch_sizes=224, overlap_ratio=0.25, padding_mode='reflect')

    reconstructed = np.zeros_like(img)
    coverage_mask = np.zeros((h, w), dtype=np.uint16)

    patches_count = 0
    start_time = time.time()

    for patch in divider.stream_patches(img):
        patches_count += 1
        md = patch.metadata
        valid_w = md.patch_size - md.pad_right
        valid_h = md.patch_size - md.pad_bottom
        patch_valid = patch.data[:valid_h, :valid_w]
        reconstructed[md.y_start:md.y_end, md.x_start:md.x_end] = patch_valid
        coverage_mask[md.y_start:md.y_end, md.x_start:md.x_end] += 1

    duration = time.time() - start_time

    assert np.all(coverage_mask >= 1), "FAIL: Some pixels are not covered!"
    assert np.array_equal(img, reconstructed), "FAIL: Reconstructed image does not match the original!"

    print(f"  Patches generated : {patches_count}")
    print(f"  Time              : {duration:.4f}s")
    print(f"  Min coverage      : {coverage_mask.min()}x")
    print(f"  Max coverage      : {coverage_mask.max()}x  (overlap regions)")
    print("[PASS] 100% spatial coverage confirmed.\n")


def test_multiscale():
    """Verifies multi-scale patch extraction produces all requested scales."""
    print("=" * 60)
    print("TEST 2: Multi-scale pipeline (64 + 128 patches)")
    print("=" * 60)
    img = create_dummy_image(300, 300)
    sizes = [64, 128]
    divider = ImageDivider(patch_sizes=sizes, overlap_ratio=0.5, padding_mode='reflect')

    scale_counts = {}
    for patch in divider.stream_patches(img):
        ps = patch.metadata.patch_size
        scale_counts[ps] = scale_counts.get(ps, 0) + 1

    assert set(scale_counts.keys()) == set(sizes), \
        f"Expected scales {set(sizes)}, got {set(scale_counts.keys())}"

    for s, c in sorted(scale_counts.items()):
        print(f"  Scale {s}×{s} : {c} patches")
    print("[PASS] All scale levels generated correctly.\n")


# ---------------------------------------------------------------------------
# Real-image test (use with your Kaggle / friend's test image)
# ---------------------------------------------------------------------------

def test_real_image(image_path: str):
    """
    Run the divider on a real image file and save sample patches for inspection.

    Args:
        image_path: Path to a JPEG / PNG / WebP image.
    """
    print("=" * 60)
    print(f"TEST 3: Real image  →  {os.path.basename(image_path)}")
    print("=" * 60)

    if not os.path.isfile(image_path):
        print(f"  [ERROR] File not found: {image_path}")
        return

    # Load the image to grab its dimensions (RGB via OpenCV)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  [ERROR] Could not decode image: {image_path}")
        return

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    file_size_kb = os.path.getsize(image_path) / 1024

    print(f"  Resolution  : {w}×{h}  ({channels} channels)")
    print(f"  File size   : {file_size_kb:.1f} KB")
    print()

    # --- Run three configurations to stress-test the divider ---
    configs = [
        {"patch_sizes": 128,        "overlap_ratio": 0.0,  "padding_mode": "reflect",  "label": "128×128, no overlap"},
        {"patch_sizes": 224,        "overlap_ratio": 0.25, "padding_mode": "reflect",  "label": "224×224, 25% overlap"},
        {"patch_sizes": [64, 128],  "overlap_ratio": 0.5,  "padding_mode": "edge",     "label": "Multi-scale 64+128, 50% overlap"},
    ]

    # Output directory for sample patches
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "divider_output")
    os.makedirs(out_dir, exist_ok=True)

    for cfg in configs:
        label = cfg.pop("label")
        divider = ImageDivider(**cfg)
        print(f"  Config: {label}")

        patch_count = 0
        coverage = np.zeros((h, w), dtype=np.uint16)
        start = time.time()

        # We'll save the first 5 patches of each config for visual inspection
        saved = 0
        max_save = 5

        for patch in divider.stream_patches(image_path):
            patch_count += 1
            md = patch.metadata
            valid_w = md.patch_size - md.pad_right
            valid_h = md.patch_size - md.pad_bottom
            coverage[md.y_start:md.y_end, md.x_start:md.x_end] += 1

            # Save a few sample patches as PNGs (BGR for OpenCV imwrite)
            if saved < max_save:
                sample = cv2.cvtColor(patch.data, cv2.COLOR_RGB2BGR)
                tag = label.replace(" ", "_").replace(",", "")
                fname = os.path.join(out_dir, f"{tag}_patch_{md.index}.png")
                cv2.imwrite(fname, sample)
                saved += 1

        elapsed = time.time() - start
        uncovered = int(np.sum(coverage == 0))

        print(f"    Patches   : {patch_count}")
        print(f"    Time      : {elapsed:.4f}s")
        print(f"    Coverage  : {'100%' if uncovered == 0 else f'{uncovered} pixels UNCOVERED!'}")
        print()

    print(f"  Sample patches saved to: {out_dir}/")
    print("[PASS] Real-image divider test complete.\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Always run internal sanity tests first
    test_coverage()
    test_multiscale()

    # If a real image path is provided, run the real-image test
    if len(sys.argv) > 1:
        test_real_image(sys.argv[1])
    else:
        print("-" * 60)
        print("TIP: To test with a real image, run:")
        print("  python3 test_divider.py path/to/image.jpg")
        print("-" * 60)
