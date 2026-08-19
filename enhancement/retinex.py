"""
Retinex-based low-light enhancement: Single-Scale Retinex (SSR) and
Multi-Scale Retinex with Color Restoration (MSRCR).

These are classical (non-learned) methods, useful as a baseline / fallback
and for quick experimentation before training Zero-DCE or MIRNet.
"""

import numpy as np
import cv2


def single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    """R(x,y) = log(I(x,y)) - log(I(x,y) * Gaussian(sigma))"""
    img = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log(img) - np.log(blur + 1.0)
    return retinex


def multi_scale_retinex(img: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)
    return retinex / len(sigmas)


def color_restoration(img: np.ndarray, alpha=125.0, beta=46.0) -> np.ndarray:
    img = img.astype(np.float32) + 1.0
    img_sum = np.sum(img, axis=2, keepdims=True)
    return beta * (np.log(alpha * img) - np.log(img_sum))


def simplest_color_balance(img: np.ndarray, low_clip=0.01, high_clip=0.99) -> np.ndarray:
    """Stretch each channel's histogram to remove outliers, restoring dynamic range."""
    total = img.shape[0] * img.shape[1]
    out_channels = []
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        unique_vals, counts = np.unique(channel, return_counts=True)
        cum = np.cumsum(counts)
        low_val = unique_vals[np.searchsorted(cum, total * low_clip)]
        high_idx = np.searchsorted(cum, total * high_clip)
        high_idx = min(high_idx, len(unique_vals) - 1)
        high_val = unique_vals[high_idx]
        channel = np.clip(channel, low_val, high_val)
        channel = (channel - low_val) / (high_val - low_val + 1e-6) * 255.0
        out_channels.append(channel)
    return np.stack(out_channels, axis=2)


def MSRCR(img: np.ndarray, sigmas=(15, 80, 250), alpha=125.0, beta=46.0,
          gain=192.0, offset=-30.0) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration.
    img: BGR uint8 image (as read by cv2.imread)
    returns: enhanced BGR uint8 image
    """
    img = img.astype(np.float32) + 1.0
    msr = multi_scale_retinex(img, sigmas)
    cr = color_restoration(img, alpha, beta)
    msrcr = gain * (msr * cr + offset)
    msrcr = simplest_color_balance(msrcr)
    msrcr = np.clip(msrcr, 0, 255).astype(np.uint8)
    return msrcr


def enhance_image_file(input_path: str, output_path: str, method: str = "msrcr"):
    """Convenience wrapper: read an image, enhance it, save it."""
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    if method == "msrcr":
        result = MSRCR(img)
    elif method == "ssr":
        r = single_scale_retinex(img.astype(np.float32), sigma=80)
        r = simplest_color_balance(r)
        result = np.clip(r, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'msrcr' or 'ssr'.")

    cv2.imwrite(output_path, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retinex low-light enhancement")
    parser.add_argument("--input", required=True, help="Path to low-light input image")
    parser.add_argument("--output", required=True, help="Path to save enhanced image")
    parser.add_argument("--method", default="msrcr", choices=["msrcr", "ssr"])
    args = parser.parse_args()
    enhance_image_file(args.input, args.output, args.method)
    print(f"Saved enhanced image to {args.output}")
