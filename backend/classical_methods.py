"""
ATLAS — Classical Thresholding Methods
Implements 5 adaptive thresholding techniques for comparison with the UNet model.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any


def apply_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu's global thresholding (1979)."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def apply_adaptive_mean(gray: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """Adaptive Mean thresholding — local neighborhood averaging."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
    )


def apply_adaptive_gaussian(gray: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """Adaptive Gaussian thresholding — Gaussian-weighted local threshold."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )


def apply_sauvola(gray: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """Sauvola thresholding (2000) — std-dev based adaptive threshold."""
    mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
    mean_sq = cv2.blur((gray.astype(np.float64)) ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    R = 128.0
    threshold = mean * (1.0 + k * (std / R - 1.0))
    mask = (gray > threshold).astype(np.uint8) * 255
    return mask


def apply_niblack(gray: np.ndarray, window_size: int = 25, k: float = -0.2) -> np.ndarray:
    """Niblack thresholding (1986) — aggressive local thresholding."""
    mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
    mean_sq = cv2.blur((gray.astype(np.float64)) ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    threshold = mean + k * std
    mask = (gray > threshold).astype(np.uint8) * 255
    return mask


METHODS = {
    "otsu": {
        "fn": apply_otsu,
        "label": "Otsu (1979)",
        "description": "Global histogram-based thresholding. Best for bright, evenly-lit scenes.",
    },
    "adaptive_mean": {
        "fn": apply_adaptive_mean,
        "label": "Adaptive Mean",
        "description": "Local neighborhood averaging. Effective in mixed lighting conditions.",
    },
    "adaptive_gaussian": {
        "fn": apply_adaptive_gaussian,
        "label": "Adaptive Gaussian",
        "description": "Gaussian-weighted local thresholding. Best for shadow-heavy scenes.",
    },
    "sauvola": {
        "fn": apply_sauvola,
        "label": "Sauvola (2000)",
        "description": "Std-dev based adaptive threshold. Ideal for low-light and night scenes.",
    },
    "niblack": {
        "fn": apply_niblack,
        "label": "Niblack (1986)",
        "description": "Aggressive local thresholding. Useful for edge and lane emphasis.",
    },
}


def preprocess_for_classical(image_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocess image for classical thresholding:
    denoise → grayscale → CLAHE contrast enhancement.
    """
    denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def run_all_methods(image_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Run all 5 classical methods on an image.

    Returns dict of method_key -> {label, description, mask (uint8), time_ms}
    """
    gray = preprocess_for_classical(image_rgb)
    results = {}
    for key, info in METHODS.items():
        t0 = time.perf_counter()
        mask = info["fn"](gray)
        elapsed = (time.perf_counter() - t0) * 1000
        results[key] = {
            "label": info["label"],
            "description": info["description"],
            "mask": mask,
            "time_ms": round(elapsed, 2),
        }
    return results


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute IoU and Dice between two binary masks (0/255 uint8).
    """
    p = (pred_mask > 127).astype(np.float32).ravel()
    g = (gt_mask > 127).astype(np.float32).ravel()

    intersection = (p * g).sum()
    union = p.sum() + g.sum() - intersection

    iou = float((intersection + 1e-6) / (union + 1e-6))
    dice = float((2.0 * intersection + 1e-6) / (p.sum() + g.sum() + 1e-6))
    accuracy = float((p == g).mean())

    return {"iou": round(iou, 4), "dice": round(dice, 4), "accuracy": round(accuracy, 4)}
