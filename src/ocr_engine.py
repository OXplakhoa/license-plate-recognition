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


def resize_for_ocr(img: np.ndarray, target_height: int = 50) -> np.ndarray:
    """Resize image to optimal height for Tesseract OCR."""
    h, w = img.shape[:2]
    if h == 0:
        return img
    
    scale = target_height / h
    new_w = int(w * scale)
    
    if new_w > 0 and target_height > 0:
        return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
    return img


def ocr_single_char(
    char_img: np.ndarray,
    whitelist: Optional[str] = None,
    lang: str = "eng",
) -> Tuple[str, float]:
    """OCR a single character image.

    Uses PSM 10 (single char) for best accuracy. Returns (char, conf).
    """
    configure_tesseract()
    
    # Resize to optimal size for character OCR
    resized = resize_for_ocr(char_img, target_height=40)
    pre = binarize_for_ocr(resized)
    pre = pad_image(pre, 10)
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
    plate_type: str | None = None,
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
    corrected = correct_plate_by_pattern(raw, plate_type=plate_type)
    return OCRResult(corrected, confs)


def ocr_plate_line(
    plate_img: np.ndarray,
    vn_plate: bool = True,
    lang: str = "eng",
    psm: int = 7,
    plate_type: str | None = None,
) -> OCRResult:
    """OCR an entire plate line using a textline PSM.

    Useful when segmentation is not available.
    """
    configure_tesseract()
    
    # Resize to optimal height
    resized = resize_for_ocr(plate_img, target_height=50)
    pre = binarize_for_ocr(resized)
    pre = pad_image(pre, 12)
    whitelist = letter_digit_whitelist(vn_plate)
    text, data = _ocr_image(pre, psm=psm, whitelist=whitelist, lang=lang, return_data=True)

    text = correct_plate_by_pattern(_normalize_plate(text), plate_type=plate_type)
    confs: List[float] = []
    if data:
        # Gather word-level confidences
        confs = [float(c) for c in data.get("conf", []) if _is_valid_conf(c)]
    return OCRResult(text, confs)


def ocr_plate_multi_psm(
    plate_img: np.ndarray,
    vn_plate: bool = True,
    lang: str = "eng",
    plate_type: str | None = None,
) -> OCRResult:
    """
    OCR plate using multiple PSM modes and return best result.
    
    Tries PSM 6, 7, 11, 13 and picks result with highest confidence
    or longest valid text.
    """
    configure_tesseract()
    
    # Resize to optimal height and preprocess
    resized = resize_for_ocr(plate_img, target_height=50)
    pre = binarize_for_ocr(resized)
    pre = pad_image(pre, 12)
    whitelist = letter_digit_whitelist(vn_plate)
    
    # Try multiple PSM modes
    psm_modes = [6, 7, 11, 13]  # Block, single line, sparse, raw line
    
    best_result = None
    best_score = -1
    
    for psm in psm_modes:
        try:
            text, data = _ocr_image(pre, psm=psm, whitelist=whitelist, lang=lang, return_data=True)
            text = correct_plate_by_pattern(_normalize_plate(text), plate_type=plate_type)
            
            confs: List[float] = []
            if data:
                confs = [float(c) for c in data.get("conf", []) if _is_valid_conf(c)]
            
            # Score based on: text length * mean confidence
            mean_conf = float(np.mean(confs)) if confs else 0
            score = len(text) * mean_conf / 100.0
            
            # Bonus for reasonable plate length (7-8 chars)
            if 7 <= len(text) <= 8:
                score += 3
            elif 5 <= len(text) <= 9:
                score += 1
            
            if score > best_score:
                best_score = score
                best_result = OCRResult(text, confs)
        except Exception:
            continue
    
    return best_result if best_result else OCRResult("", [])


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
        'O': '0', 'D': '0', 'Q': '0', 'o': '0', 'C': '0',
        'I': '1', 'l': '1', 'T': '1', 'i': '1', 'J': '1',
        'Z': '2', 'z': '2',
        'E': '3',
        'A': '4', 'H': '4',
        'S': '5', 's': '5',
        'b': '6', 'G': '6', 'g': '6',
        'B': '8', 'R': '8',
        'g': '9', 'q': '9',
    }
    return mapping.get(ch, ch) if ch.isalpha() else ch


def _closest_letter(ch: str) -> str:
    # For the serial letter, prefer letters over digits; map obvious digit confusions
    mapping = {
        '0': 'D', '1': 'T', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '9': 'G',
        '4': 'A', '3': 'E',
    }
    return mapping.get(ch, ch)


# Vietnamese province codes (first 2 digits)
VN_PROVINCE_CODES = {
    '11': 'Cao Bằng', '12': 'Lạng Sơn', '14': 'Quảng Ninh', '15': 'Hải Phòng',
    '16': 'Hải Phòng', '17': 'Thái Bình', '18': 'Nam Định', '19': 'Phú Thọ',
    '20': 'Thái Nguyên', '21': 'Yên Bái', '22': 'Tuyên Quang', '23': 'Hà Giang',
    '24': 'Lào Cai', '25': 'Lai Châu', '26': 'Sơn La', '27': 'Điện Biên',
    '28': 'Hoà Bình', '29': 'Hà Nội', '30': 'Hà Nội', '31': 'Hà Nội',
    '32': 'Hà Nội', '33': 'Hà Nội', '34': 'Hải Dương', '35': 'Ninh Bình',
    '36': 'Thanh Hoá', '37': 'Nghệ An', '38': 'Hà Tĩnh', '39': 'Đồng Nai',
    '40': 'Hà Nội', '41': 'Hà Nội', '43': 'Đà Nẵng', '47': 'Đắk Lắk',
    '48': 'Đắk Nông', '49': 'Lâm Đồng', '50': 'TP.HCM', '51': 'TP.HCM',
    '52': 'TP.HCM', '53': 'TP.HCM', '54': 'TP.HCM', '55': 'TP.HCM',
    '56': 'TP.HCM', '57': 'TP.HCM', '58': 'TP.HCM', '59': 'TP.HCM',
    '60': 'Đồng Nai', '61': 'Bình Dương', '62': 'Long An', '63': 'Tiền Giang',
    '64': 'Vĩnh Long', '65': 'Cần Thơ', '66': 'Đồng Tháp', '67': 'An Giang',
    '68': 'Kiên Giang', '69': 'Cà Mau', '70': 'Tây Ninh', '71': 'Bến Tre',
    '72': 'Bà Rịa - VT', '73': 'Quảng Bình', '74': 'Quảng Trị', '75': 'Huế',
    '76': 'Quảng Ngãi', '77': 'Bình Định', '78': 'Phú Yên', '79': 'Khánh Hoà',
    '81': 'Gia Lai', '82': 'Kon Tum', '83': 'Sóc Trăng', '84': 'Trà Vinh',
    '85': 'Ninh Thuận', '86': 'Bình Thuận', '88': 'Vĩnh Phúc', '89': 'Hưng Yên',
    '90': 'Hà Nam', '92': 'Quảng Nam', '93': 'Bình Phước', '94': 'Bạc Liêu',
    '95': 'Hậu Giang', '97': 'Bắc Kạn', '98': 'Bắc Giang', '99': 'Bắc Ninh',
}

# Valid serial letters (position 3)
VN_SERIAL_LETTERS = set('ABCDEFGHKLMNPRSTUVXYZ')  # No I, O, Q, J, W


def validate_vn_plate_format(text: str) -> Tuple[bool, str]:
    """
    Validate if text matches Vietnamese plate format.
    
    Format: NN-C-NNNNN or NNC-NNNNN (7-8 chars)
    - NN: Province code (2 digits)
    - C: Serial letter
    - NNNNN: 4-5 digits
    
    Returns:
        (is_valid, reason)
    """
    if len(text) < 7 or len(text) > 9:
        return False, f"Length {len(text)} not in range [7,9]"
    
    # Check province code
    province = text[:2]
    if not province.isdigit():
        return False, f"Province code '{province}' not digits"
    if province not in VN_PROVINCE_CODES:
        return False, f"Unknown province code '{province}'"
    
    # Check serial letter
    if len(text) > 2:
        serial = text[2]
        if not serial.isalpha():
            return False, f"Serial '{serial}' not letter"
        if serial not in VN_SERIAL_LETTERS:
            return False, f"Serial '{serial}' not valid VN letter"
    
    # Check remaining digits
    remaining = text[3:]
    if not remaining.isdigit():
        return False, f"Remaining '{remaining}' not all digits"
    
    return True, "Valid"


def correct_plate_by_pattern(text: str, plate_type: str | None = None) -> str:
    """Normalize OCR output to common VN plate patterns.

    Applies digit/letter lookalike normalization and trims to typical length
    (7–8 chars). plate_type can be "car1", "car2", or "bike"; all share the
    two-digit prefix + one letter convention.
    """
    t = list(_normalize_plate(text))
    if len(t) < 3:
        return "".join(t)

    # positions 0,1 digits
    for pos in [0, 1]:
        if pos < len(t):
            t[pos] = _closest_digit(t[pos])
            if not t[pos].isdigit():
                t[pos] = '0'

    # position 2 letter
    if 2 < len(t):
        c = _closest_letter(t[2])
        if c.isdigit() or c in {'I', 'O', 'Q'}:
            c = 'A'
        t[2] = c

    # remaining digits
    for pos in range(3, len(t)):
        t[pos] = _closest_digit(t[pos])
        if not t[pos].isdigit():
            t[pos] = '0'

    normalized = "".join(t)
    # Trim overly long strings to 8 characters (common for VN plates)
    if len(normalized) > 8 and (plate_type in {"car1", "car2", "bike", None}):
        normalized = normalized[:8]
    return normalized


def smart_correct_plate(text: str, plate_type: str | None = None) -> Tuple[str, float]:
    """
    Smart plate correction with confidence score.
    
    Uses province code validation and pattern matching for better accuracy.
    
    Args:
        text: Raw OCR text
        plate_type: Optional plate type hint
        
    Returns:
        (corrected_text, confidence_score)
    """
    # First apply basic correction
    corrected = correct_plate_by_pattern(text, plate_type=plate_type)
    
    # Validate format
    is_valid, reason = validate_vn_plate_format(corrected)
    
    if is_valid:
        return corrected, 1.0
    
    # Try to fix province code
    if len(corrected) >= 2:
        province = corrected[:2]
        if province not in VN_PROVINCE_CODES:
            # Try common corrections
            corrections = {
                '00': '50', '01': '51', '10': '10',
                '0D': '50', '5D': '50', 'S0': '50',
                'S1': '51', 'S2': '52', 'S9': '59',
            }
            if province in corrections:
                corrected = corrections[province] + corrected[2:]
    
    # Re-validate
    is_valid, _ = validate_vn_plate_format(corrected)
    confidence = 0.8 if is_valid else 0.5
    
    return corrected, confidence


def format_plate_display(text: str) -> str:
    """
    Format plate text for display with common VN separators.
    
    Input: "51F12345"
    Output: "51F-123.45"
    """
    if len(text) < 7:
        return text
    
    # Format: NN-C-NNN.NN
    province = text[:2]
    serial = text[2]
    numbers = text[3:]
    
    if len(numbers) >= 5:
        return f"{province}{serial}-{numbers[:3]}.{numbers[3:]}"
    elif len(numbers) >= 3:
        return f"{province}{serial}-{numbers}"
    else:
        return text
