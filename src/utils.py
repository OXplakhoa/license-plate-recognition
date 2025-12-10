from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Ensure image is single-channel grayscale (uint8)."""
    if img is None:
        raise ValueError("Input image is None")
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def binarize_for_ocr(img: np.ndarray, invert_pref: bool = True, sharpen: bool = True) -> np.ndarray:
    """Binarize image for Tesseract: black text on white background.

    Steps:
    - grayscale
    - CLAHE for local contrast
    - Sharpen (optional)
    - Otsu threshold
    - invert if text likely white on dark background
    """
    gray = ensure_grayscale(img)

    # Local contrast enhancement helps in varied lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Sharpen for better character edges
    if sharpen:
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel_sharpen)

    # Otsu threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if invert_pref:
        # Make text black on white (Tesseract prefers this)
        # Heuristic: if black pixels are minority assume current is black text.
        black_ratio = (th == 0).mean()
        # If black area is large (>0.6), assume background is black => invert
        if black_ratio > 0.6:
            th = cv2.bitwise_not(th)
    else:
        # Always force black text on white background
        # If currently white text on black, invert
        if (th == 255).mean() < 0.5:
            th = cv2.bitwise_not(th)

    return th


def pad_image(img: np.ndarray, pad: int = 10, value: int = 255) -> np.ndarray:
    """Pad image with white border to give Tesseract context."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)


def get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable, supporting .env if python-dotenv is installed."""
    # Lazy import to avoid hard dependency if user prefers OS env
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        # dotenv not installed: rely on OS env
        pass
    return os.getenv(key, default)


def letter_digit_whitelist(vn_plate: bool = True) -> str:
    """Return a safe whitelist for VN license plates.

    Typical VN plates avoid ambiguous letters: I, O, Q.
    """
    digits = "0123456789"
    letters = "ABCDEFGHJKLMNPRSTUVWXZ"  # exclude I, O, Q, Y is often avoided but leave X/Z
    return digits + letters if vn_plate else digits + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
