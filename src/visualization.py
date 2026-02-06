"""
Visualization helpers for the license-plate recognition pipeline.

Pure image-processing functions that produce intermediate step images
for educational / debugging display.  **No Streamlit dependency** –
all functions accept and return NumPy arrays / plain dicts.
"""

import cv2
import numpy as np

from src.lp_detector import compute_character_score


# ---------------------------------------------------------------------------
# Full-image preprocessing visualization
# ---------------------------------------------------------------------------

def visualize_preprocessing_steps(image: np.ndarray) -> dict:
    """
    Run each preprocessing stage on *image* (BGR) and return a dict
    of intermediate results suitable for UI display.

    Keys returned
    -------------
    grayscale, blurred, clahe, edges, morphology        – grayscale images
    contours, candidates, scored_candidates              – BGR annotated images
    contour_count, candidate_count                       – int counts
    plate_candidates                                     – list[(x,y,w,h,ar)]
    scored_list                                          – list[dict]
    """
    steps: dict = {}

    # Step 1 – Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps["grayscale"] = gray

    # Step 2 – Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    steps["blurred"] = blurred

    # Step 3 – CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    steps["clahe"] = enhanced

    # Step 4 – Canny Edge Detection
    edges = cv2.Canny(enhanced, 50, 150)
    steps["edges"] = edges

    # Step 5 – Morphological close + open
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open)
    steps["morphology"] = morph

    # Step 6 – Find & draw contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    steps["contours"] = contour_img
    steps["contour_count"] = len(contours)

    # Step 7 – Filter by aspect-ratio / area
    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img
    valid_contours: list = []
    plate_candidates: list = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        if 0.005 < area / img_area < 0.3 and 0.5 < aspect_ratio < 6.0:
            valid_contours.append(cnt)
            plate_candidates.append((x, y, w, h, aspect_ratio))

    candidate_img = image.copy()
    for cnt in valid_contours:
        cv2.drawContours(candidate_img, [cnt], -1, (0, 255, 0), 2)
    for (x, y, w, h, ar) in plate_candidates:
        cv2.rectangle(candidate_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            candidate_img, f"AR:{ar:.1f}", (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
        )

    steps["candidates"] = candidate_img
    steps["candidate_count"] = len(plate_candidates)
    steps["plate_candidates"] = plate_candidates

    # Step 8 – Character-score validation
    scored_candidates = _score_candidates(gray, plate_candidates)

    scored_img = image.copy()
    for cand in scored_candidates:
        x, y, w, h = cand["box"]
        color = (0, 255, 0) if cand["is_valid"] else (0, 0, 255)
        cv2.rectangle(scored_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            scored_img, f"S:{cand['char_score']:.2f}", (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    steps["scored_candidates"] = scored_img
    steps["scored_list"] = scored_candidates

    return steps


# ---------------------------------------------------------------------------
# Plate-ROI processing visualization
# ---------------------------------------------------------------------------

def visualize_plate_processing(plate_roi: np.ndarray) -> dict:
    """
    Process a cropped plate ROI (BGR or grayscale) through binarization
    and character segmentation, returning intermediate images.

    Keys returned
    -------------
    gray, resized, clahe, otsu, cleaned     – grayscale images
    segmented                                – BGR annotated image
    char_count                               – int
    char_boxes                               – list[(x,y,w,h)]
    char_images_28x28                        – list[np.ndarray]  (28×28 uint8)
    """
    steps: dict = {}

    # Grayscale
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY) if plate_roi.ndim == 3 else plate_roi.copy()
    steps["gray"] = gray

    # Resize small plates
    h, w = gray.shape[:2]
    if h < 50:
        scale = 50 / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    steps["resized"] = gray

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    steps["clahe"] = enhanced

    # Otsu binarization
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps["otsu"] = binary

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps["cleaned"] = cleaned

    # Character contour extraction
    char_contours = _extract_char_contours(cleaned)

    # Annotated segmentation image
    segmented_img = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(char_contours):
        color = (0, 255, 0) if i % 2 == 0 else (0, 200, 255)
        cv2.rectangle(segmented_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(segmented_img, str(i + 1), (x, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    steps["segmented"] = segmented_img
    steps["char_count"] = len(char_contours)
    steps["char_boxes"] = char_contours

    # 28×28 MNIST-style character images
    steps["char_images_28x28"] = _extract_char_images_28x28(cleaned, char_contours)

    return steps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_candidates(gray: np.ndarray, plate_candidates: list) -> list:
    """Score each candidate region with ``compute_character_score``."""
    scored: list = []
    for i, (x, y, w, h, ar) in enumerate(plate_candidates):
        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        score, details = compute_character_score(roi)
        scored.append({
            "index": i + 1,
            "box": (x, y, w, h),
            "aspect_ratio": ar,
            "char_score": score,
            "char_count": details.get("char_count", 0),
            "is_valid": score >= 0.35,
        })
    scored.sort(key=lambda c: c["char_score"], reverse=True)
    return scored


def _extract_char_contours(cleaned: np.ndarray) -> list:
    """Find character-sized contours sorted left-to-right."""
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_h = cleaned.shape[0]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.2 * char_h < h < 0.95 * char_h and w > 3:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda c: c[0])
    return boxes


def _extract_char_images_28x28(cleaned: np.ndarray, char_contours: list) -> list:
    """Extract each character, pad to square, resize to 28×28 (white-on-black)."""
    inverted = cv2.bitwise_not(cleaned)
    images: list = []
    for x, y, w, h in char_contours:
        char_img = inverted[y : y + h, x : x + w]
        side = max(h, w) + 4
        canvas = np.zeros((side, side), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        canvas[y_off : y_off + h, x_off : x_off + w] = char_img
        images.append(cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA))
    return images
