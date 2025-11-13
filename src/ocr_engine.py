from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .utils import binarize_for_ocr, pad_image, get_env, letter_digit_whitelist, ensure_grayscale


@dataclass
class OCRResult:
    text: str
    confidences: List[float]

    @property
    def mean_conf(self) -> float:
        vals = [c for c in self.confidences if c >= 0]
        return float(np.mean(vals)) if vals else -1.0


def configure_tesseract() -> None:
    """Configure pytesseract using .env if available.

    Honors TESSERACT_PATH and TESSERACT_DATA_PATH when present.
    """
    tesseract_cmd = get_env("TESSERACT_PATH")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    tessdata_dir = get_env("TESSERACT_DATA_PATH")
    if tessdata_dir and os.path.isdir(tessdata_dir):
        # Tesseract expects TESSDATA_PREFIX to point to the tessdata directory
        os.environ.setdefault("TESSDATA_PREFIX", tessdata_dir)


def _ocr_image(
    img: np.ndarray,
    psm: int,
    whitelist: Optional[str] = None,
    lang: str = "eng",
    return_data: bool = False,
) -> Tuple[str, Optional[dict]]:
    """Internal helper to OCR an image with config.

    Returns the recognized string and optionally the full data dict.
    """
    config_parts = [f"--psm {psm}", "-l", lang, "--oem 3"]
    if whitelist:
        config_parts += [f"-c tessedit_char_whitelist={whitelist}"]

    # Avoid passing --tessdata-dir; rely on TESSDATA_PREFIX to reduce quoting/splitting issues

    config = " ".join(config_parts)

    if return_data:
        data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
        text = pytesseract.image_to_string(img, config=config)
        return text, data
    else:
        text = pytesseract.image_to_string(img, config=config)
        return text, None


def ocr_single_char(
    char_img: np.ndarray,
    whitelist: Optional[str] = None,
    lang: str = "eng",
) -> Tuple[str, float]:
    """OCR a single character image.

    Uses PSM 10 (single char) for best accuracy. Returns (char, conf).
    """
    configure_tesseract()
    pre = binarize_for_ocr(char_img)
    pre = pad_image(pre, 8)
    _, data = _ocr_image(pre, psm=10, whitelist=whitelist, lang=lang, return_data=True)

    # Find the highest-confidence non-empty symbol
    best_char = ""
    best_conf = -1.0
    if data:
        for txt, conf in zip(data.get("text", []), data.get("conf", [])):
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                conf_val = -1.0
            t = (txt or "").strip()
            if t and conf_val > best_conf:
                # choose the most confident single character (first char if word)
                best_char = t[0]
                best_conf = conf_val

    # Fallback: if data didn't return, try raw string
    if not best_char:
        txt = pytesseract.image_to_string(pre, config=f"--psm 10 -l {lang}").strip()
        best_char = txt[:1] if txt else ""
        best_conf = -1.0

    return best_char.upper(), best_conf


def ocr_characters(
    char_images: Sequence[np.ndarray],
    vn_plate: bool = True,
    lang: str = "eng",
) -> OCRResult:
    """OCR a list of segmented character images and join them into a string.

    - Applies char whitelist suitable for VN plates
    - Returns joined text and list of confidences
    """
    whitelist = letter_digit_whitelist(vn_plate)
    chars: List[str] = []
    confs: List[float] = []
    for i, img in enumerate(char_images):
        ch, cf = ocr_single_char(img, whitelist=whitelist, lang=lang)
        if ch:
            chars.append(ch)
            confs.append(cf)
    raw = "".join(chars)
    corrected = correct_plate_by_pattern(raw)
    return OCRResult(corrected, confs)


def ocr_plate_line(
    plate_img: np.ndarray,
    vn_plate: bool = True,
    lang: str = "eng",
    psm: int = 7,
) -> OCRResult:
    """OCR an entire plate line using a textline PSM.

    Useful when segmentation is not available.
    """
    configure_tesseract()
    pre = binarize_for_ocr(plate_img)
    pre = pad_image(pre, 8)
    whitelist = letter_digit_whitelist(vn_plate)
    text, data = _ocr_image(pre, psm=psm, whitelist=whitelist, lang=lang, return_data=True)

    text = correct_plate_by_pattern(_normalize_plate(text))
    confs: List[float] = []
    if data:
        # Gather word-level confidences
        confs = [float(c) for c in data.get("conf", []) if _is_valid_conf(c)]
    return OCRResult(text, confs)


def _is_valid_conf(c: object) -> bool:
    try:
        return float(c) >= 0
    except (TypeError, ValueError):
        return False


def _normalize_plate(text: str) -> str:
    # Keep only digits and A-Z, uppercase
    clean = []
    for ch in (text or "").upper():
        if ch.isdigit() or ("A" <= ch <= "Z"):
            clean.append(ch)
    return "".join(clean)


def combine_lines(line1: str, line2: Optional[str] = None) -> str:
    """Combine one or two lines into a single plate string without separators."""
    if line2:
        return _normalize_plate(line1) + _normalize_plate(line2)
    return _normalize_plate(line1)


def _closest_digit(ch: str) -> str:
    # Common OCR confusions for digits
    mapping = {
        'O': '0', 'D': '0', 'Q': '0', 'o': '0',
        'I': '1', 'l': '1', 'T': '1',
        'Z': '2',
        'S': '5',
        'b': '6', 'G': '6',
        'B': '8',
    }
    return mapping.get(ch, ch) if ch.isalpha() else ch


def _closest_letter(ch: str) -> str:
    # For the serial letter, prefer letters over digits; map obvious digit confusions
    mapping = {
        '0': 'O', '1': 'T', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'
    }
    return mapping.get(ch, ch)


def correct_plate_by_pattern(text: str) -> str:
    """Heuristically correct OCR output to match VN plate pattern: NN L DDDDD

    - First two: digits
    - Third: letter
    - Rest: digits
    Applies digit/letter lookalike normalization.
    """
    t = list(_normalize_plate(text))
    if len(t) < 7:
        return "".join(t)
    # positions 0,1 digits
    for pos in [0, 1]:
        if pos < len(t):
            t[pos] = _closest_digit(t[pos])
            if not t[pos].isdigit():
                # if still not a digit, coerce to '0'
                t[pos] = '0'
    # position 2 letter
    if 2 < len(t):
        c = _closest_letter(t[2])
        # Avoid ambiguous letters and digits
        if c.isdigit() or c in {'I', 'O', 'Q'}:
            c = 'A'
        t[2] = c
    # remaining digits
    for pos in range(3, len(t)):
        t[pos] = _closest_digit(t[pos])
        if not t[pos].isdigit():
            t[pos] = '0'
    return "".join(t)
