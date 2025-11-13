import os
from pathlib import Path

import cv2
import numpy as np

import pytest
import pytesseract

from src.ocr_engine import ocr_characters, OCRResult, configure_tesseract
from src.utils import letter_digit_whitelist


def _render_char_image(ch: str, font_scale: float = 2.0) -> np.ndarray:
    # Create a white canvas and render a single black character centered
    canvas = np.full((80, 60), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3
    size, _ = cv2.getTextSize(ch, font, font_scale, thickness)
    text_w, text_h = size
    org = ((canvas.shape[1] - text_w) // 2, (canvas.shape[0] + text_h) // 2 - 5)
    cv2.putText(canvas, ch, org, font, font_scale, (0,), thickness, cv2.LINE_AA)
    return canvas


def test_per_character_ocr_for_known_plate():
    # Ensure tesseract is configured from .env if present
    configure_tesseract()
    # Skip test gracefully if Tesseract is not available on this machine
    try:
        # Quick probe: run OCR on a tiny synthetic image to ensure tessdata is available
        probe = _render_char_image('A')
        _ = pytesseract.image_to_string(probe, config="--psm 10 -l eng")
    except Exception:
        pytest.skip("Tesseract or tessdata not available; skipping OCR test.")

    plate = "30F78286"
    imgs = [_render_char_image(c) for c in plate]

    result: OCRResult = ocr_characters(imgs, vn_plate=True, lang="eng")

    assert result.text == plate, f"Expected {plate}, got {result.text}"
    # Average confidence should be decent on synthetic render
    assert result.mean_conf == -1.0 or result.mean_conf >= 60.0
