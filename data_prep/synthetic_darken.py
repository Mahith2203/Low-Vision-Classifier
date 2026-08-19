"""
Synthetically darkens a normal-light dataset to create low-light training data.
Useful when you don't have real paired low/normal-light images, or want to
supplement ExDark/LOL with a larger, more diverse dataset (e.g. COCO, VOC,
or your own labeled bright-light images).

Two modes:
  1. "pairs"    -> darkens images and keeps the originals, producing a PAIRED
                   low/high dataset for MIRNet training.
  2. "darken_only" -> just darkens images in place (e.g. to feed detection/
                   classification pipelines that need dark inputs but don't
                   need paired ground truth).

Degradation model (standard in low-light literature):
    1. Gamma correction (I_dark = I^gamma, gamma > 1 darkens the image)
    2. Random linear scaling (simulates underexposure)
    3. Additive Gaussian + Poisson-like sensor noise (simulates low-light sensor noise)
    4. Optional slight Gaussian blur (simulates motion blur / high ISO softness)

Usage:
    # Paired mode (for MIRNet): produces data/synthetic_low/{low,high}/
    python synthetic_darken.py --input_dir ./data/bright_images --output_dir ./data/synthetic_low \
        --mode pairs --gamma_range 2.0 3.5 --noise_std 0.02

    # Darken-only mode (for detection/classification, labels untouched): overwrites images,
    # labels are pixel-independent (bounding boxes / class folders don't need to change)
    python synthetic_darken.py --input_dir ./data/coco_bright --output_dir ./data/coco_dark \
        --mode darken_only --gamma_range 1.8 3.0 --noise_std 0.02
"""

import os
import argparse
import random
import glob
import numpy as np
from PIL import Image


def darken_image(img: np.ndarray, gamma_range=(1.8, 3.0), scale_range=(0.3, 0.6),
                  noise_std=0.02, blur_prob=0.3) -> np.ndarray:
    """
    img: float32 array, HxWx3, values in [0, 1]
    returns: darkened float32 array, values in [0, 1]
    """
    gamma = random.uniform(*gamma_range)
    scale = random.uniform(*scale_range)

    dark = np.power(img, gamma) * scale

    # sensor noise: signal-dependent (more noise in darker regions, like real low-light sensors)
    noise = np.random.normal(0, noise_std, dark.shape).astype(np.float32)
    dark = dark + noise * (1.0 - dark)  # more noise where image is darker

    if random.random() < blur_prob:
        from scipy.ndimage import gaussian_filter
        sigma = random.uniform(0.4, 1.0)
        dark = gaussian_filter(dark, sigma=(sigma, sigma, 0))

    dark = np.clip(dark, 0.0, 1.0)
    return dark


def process_directory(input_dir, output_dir, mode, gamma_range, scale_range, noise_std, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    print(f"Found {len(image_paths)} images in {input_dir}")

    if mode == "pairs":
        low_dir = os.path.join(output_dir, "low")
        high_dir = os.path.join(output_dir, "high")
        os.makedirs(low_dir, exist_ok=True)
        os.makedirs(high_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0

        dark_arr = darken_image(arr, gamma_range, scale_range, noise_std)
        dark_img = Image.fromarray((dark_arr * 255).astype(np.uint8))

        fname = os.path.basename(path)

        if mode == "pairs":
            dark_img.save(os.path.join(low_dir, fname))
            img.save(os.path.join(high_dir, fname))  # keep original as "high" ground truth
        else:
            # preserve relative subfolder structure (important for ImageFolder-style class dirs)
            rel_path = os.path.relpath(path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            dark_img.save(out_path)

        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(image_paths)}")

    print(f"Done. Output written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder of normal-light images (subfolders OK)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["pairs", "darken_only"], default="pairs")
    parser.add_argument("--gamma_range", type=float, nargs=2, default=(1.8, 3.0),
                         help="Higher gamma = darker. Try 1.5-2.5 for mild, 2.5-4.0 for very dark.")
    parser.add_argument("--scale_range", type=float, nargs=2, default=(0.3, 0.6))
    parser.add_argument("--noise_std", type=float, default=0.02)
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.mode,
                       tuple(args.gamma_range), tuple(args.scale_range), args.noise_std)
