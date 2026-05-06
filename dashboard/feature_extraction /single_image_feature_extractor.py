"""
Single Image Feature Extractor
===============================
Extracts RGB, Gabor, and Sobel features from a single image and bounding box,
using the corrected parameters from fix_features.py.

Usage
-----
    python single_image_feature_extractor.py \
        --image_path /path/to/image.jpg \
        --bbox_file /path/to/bbox.txt \
        --output_dir /path/to/output

bbox.txt should contain: xmin ymin xmax ymax (space-separated integers)
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# FIXED CONSTANTS (from fix_features.py)
# ---------------------------------------------------------------------------
TARGET_H, TARGET_W = 64, 64
N_NODES = TARGET_H * TARGET_W

# Gabor — matching baseline_handover.py (sigma=lambda/pi)
GABOR_LAMBDA = 6.0
GABOR_FREQ = 1.0 / GABOR_LAMBDA
GABOR_GAMMA = 1.0
GABOR_SIGMA = GABOR_LAMBDA / np.pi  # ≈ 1.91
GABOR_THETAS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GABOR_PSI = 0

_ks = int(np.ceil(6 * GABOR_SIGMA))
GABOR_KSIZE = _ks if _ks % 2 == 1 else _ks + 1


# ===========================================================================
# IMAGE CROP + RESIZE (from please_work.py and fix_features.py)
# ===========================================================================

def crop_and_resize(img_bgr: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int) -> np.ndarray:
    """
    Crop the bounding box from the full image and resize to 64×64.
    Clamps coordinates to valid image bounds before slicing.
    """
    h, w = img_bgr.shape[:2]
    xmin = max(0, xmin);  ymin = max(0, ymin)
    xmax = min(w, xmax);  ymax = min(h, ymax)

    if xmax <= xmin or ymax <= ymin:
        # Degenerate box — return a black patch
        return np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)

    crop = img_bgr[ymin:ymax, xmin:xmax]
    resized = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    return resized   # uint8, BGR, shape (64, 64, 3)


# ===========================================================================
# FEATURE EXTRACTION (FIXED from fix_features.py)
# ===========================================================================

def extract_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """RGB extraction (from please_work.py)"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.reshape(N_NODES, 3)


# Build Gabor kernels once (FIXED σ)
def _build_gabor_kernels() -> list[np.ndarray]:
    kernels = []
    for theta in GABOR_THETAS:
        k = cv2.getGaborKernel(
            ksize=(GABOR_KSIZE, GABOR_KSIZE),
            sigma=GABOR_SIGMA,
            theta=theta,
            lambd=GABOR_LAMBDA,
            gamma=GABOR_GAMMA,
            psi=GABOR_PSI,
            ktype=cv2.CV_32F,
        )
        kernels.append(k)
    return kernels

GABOR_KERNELS = _build_gabor_kernels()


def extract_gabor(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gabor extraction with per-channel min-max normalization (matching training data).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    channels = []
    for k in GABOR_KERNELS:
        resp = cv2.filter2D(gray, cv2.CV_32F, k)
        channels.append(resp.ravel())

    raw = np.stack(channels, axis=1).astype(np.float32)

    gmin = raw.min(axis=0, keepdims=True)
    gmax = raw.max(axis=0, keepdims=True)
    denom = gmax - gmin
    denom[denom == 0] = 1.0
    return (raw - gmin) / denom


def extract_sobel(img_bgr: np.ndarray) -> np.ndarray:
    """
    Sobel extraction with per-sample min-max normalization (matching training data).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    raw = magnitude.ravel().reshape(N_NODES, 1).astype(np.float32)

    smin = raw.min()
    smax = raw.max()
    denom = smax - smin
    if denom == 0:
        denom = 1.0
    return (raw - smin) / denom


# ===========================================================================
# MAIN SCRIPT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract features from a single image and bounding box")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--bbox_file", type=str, required=True, help="Path to the bounding box file (xmin ymin xmax ymax)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save features")

    args = parser.parse_args()

    # Load image
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image from {args.image_path}")

    # Load bounding box
    with open(args.bbox_file, 'r') as f:
        bbox_line = f.read().strip()
        xmin, ymin, xmax, ymax = map(int, bbox_line.split())

    # Crop and resize
    cropped = crop_and_resize(img_bgr, xmin, ymin, xmax, ymax)

    # Extract features
    rgb_feat = extract_rgb(cropped)
    gabor_feat = extract_gabor(cropped)
    sobel_feat = extract_sobel(cropped)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    np.save(output_dir / "rgb.npy", rgb_feat)
    np.save(output_dir / "gabor.npy", gabor_feat)
    np.save(output_dir / "sobel.npy", sobel_feat)

    print(f"Features saved to {output_dir}")
    print(f"RGB shape: {rgb_feat.shape}")
    print(f"Gabor shape: {gabor_feat.shape}")
    print(f"Sobel shape: {sobel_feat.shape}")


if __name__ == "__main__":
    main()