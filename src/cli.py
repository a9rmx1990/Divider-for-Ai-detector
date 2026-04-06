import sys
import os
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path

# ── Auto-bootstrap: inject venv site-packages if module not found ──
def _ensure_scanner_forensics():
    """Import scanner_forensics, auto-injecting the venv if needed."""
    try:
        from scanner_forensics import ImageAnalyzer
        return ImageAnalyzer
    except ImportError:
        pass

    # Locate the project venv relative to this script (up one directory from src/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    venv_site = project_root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

    if not venv_site.exists():
        print("✗ Scanner Forensics library not found!")
        print("  Virtual environment not set up. Run:")
        print(f"    cd {project_root}")
        print("    python3 -m venv .venv")
        print("    source .venv/bin/activate")
        print("    pip install maturin numpy Pillow")
        print("    cd rust-div && maturin develop --release")
        sys.exit(1)

    # Inject venv site-packages into sys.path and retry
    sys.path.insert(0, str(venv_site))

    try:
        from scanner_forensics import ImageAnalyzer
        return ImageAnalyzer
    except ImportError:
        print("✗ Scanner Forensics library not found inside the venv!")
        print("  Rebuild it:")
        print(f"    cd {project_root}")
        print("    source .venv/bin/activate")
        print("    cd rust-div && maturin develop --release")
        sys.exit(1)


def detect_ai(image_path: str):
    """Analyze image using the Rust scanner_forensics library."""
    ImageAnalyzer = _ensure_scanner_forensics()

    print(f"\n📊 Analyzing {image_path}...")
    
    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        sys.exit(1)

    h, w, c = img_array.shape
    
    # Initialize analyzer
    analyzer = ImageAnalyzer(patch_size=256, stride=128)
    
    # Get stats
    stats = analyzer.get_patch_stats(h, w)
    print(f"  Grid plan: {int(stats['total_patches'])} patches ({int(stats['num_rows'])}×{int(stats['num_cols'])})")
    
    # Analyze image
    results = analyzer.analyze_image(img_array, str(image_path), extract_exif=False)

    # Interpret results
    hf_lf_values = [p.hf_lf_ratio for p in results]
    flatness_values = [p.spectral_flatness for p in results]
    
    if not hf_lf_values:
        print("No patches could be extracted.")
        sys.exit(1)
        
    hf_lf_mean = np.mean(hf_lf_values)
    flatness_mean = np.mean(flatness_values)
    anomaly_count = sum(1 for p in results if p.anomaly_detected)
    anomaly_ratio = anomaly_count / len(results)

    print(f"\n🔍 Detection Results ({len(results)} patches):")
    print("-" * 50)
    print(f"  HF/LF Avg Ratio: {hf_lf_mean:.4f}")
    print(f"  Spectral Flatness: {flatness_mean:.4f}")
    print(f"  Anomalous Patches: {anomaly_count} / {len(results)}")

    # Detection logic:
    # Mathematical Z-Score Peak Detection: The primary signal is now the anomaly_ratio,
    # which accurately counts patches containing unnatural 5-sigma high-frequency 
    # periodic spikes (checkerboard artifacts from Conv/Diffusion models).
    score = 0.0
    
    # 1. Structural Anomaly Spike Detection (Most robust mathematical signal)
    if anomaly_ratio > 0.15:
        score += 3.0
    elif anomaly_ratio > 0.05:
        score += 2.0
    elif anomaly_ratio > 0.0:
        score += 1.0

    # 2. Spectral Baseline features (smoothness / lack of camera shot noise)
    if flatness_mean < 0.26:
        score += 1.0
    elif flatness_mean < 0.29:
        score += 0.5

    if hf_lf_mean < 0.035:
        score += 1.0
    elif hf_lf_mean < 0.043:
        score += 0.5

    if score < 1.0:
        verdict = "🟢 LIKELY REAL"
        confidence = max(60.0, 99.0 - (score * 15))
    elif score < 2.0:
        verdict = "🟡 UNCERTAIN"
        confidence = 50.0
    elif score < 3.0:
        verdict = "🟠 SUSPICIOUS"
        confidence = min(85.0, 50.0 + (score * 10))
    else:
        verdict = "🔴 LIKELY AI-GENERATED"
        confidence = min(99.0, 70.0 + (score * 5))

    print(f"\n🤖 FINAL VERDICT: {verdict}")
    print(f"  Confidence: {confidence:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Name of the image should be provided")
        sys.exit(1)
        
    target_image = sys.argv[1]
    detect_ai(target_image)
