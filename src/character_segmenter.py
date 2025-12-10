from __future__ import annotations

"""
Character segmentation utilities for Vietnamese license plates.
Simplified from the exploration notebook but keeps key heuristics for
2-line car plates, 1-line rectangular car plates, and motorbike plates.

Enhanced for perspective-corrected plates with better noise handling.
"""
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from .utils import ensure_grayscale


@dataclass
class SegmentationResult:
    boxes: List[Tuple[int, int, int, int]]
    char_images: List[np.ndarray]
    inverted_binary: np.ndarray
    binary: np.ndarray
    debug_info: Dict[str, Any] = field(default_factory=dict)


def binarize_plate(
    plate_roi: np.ndarray, 
    method: str = "otsu",
    denoise: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binarize plate image for character segmentation.
    
    Args:
        plate_roi: Input plate ROI (grayscale or BGR)
        method: "otsu", "adaptive", or "combined"
        denoise: Apply denoising before binarization
        
    Returns:
        (binary, inverted_binary) tuple
    """
    gray = ensure_grayscale(plate_roi)
    
    # Apply CLAHE for contrast enhancement - increased clipLimit for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    
    # Optional denoising for corrected plates (may have interpolation artifacts)
    if denoise:
        denoised = cv2.fastNlMeansDenoising(clahe, None, h=8, templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = clahe
    
    # Sharpen the image for better character edges
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
    
    blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)  # Reduced blur kernel

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
    elif method == "combined":
        # Try both methods and choose better one based on character-like regions
        _, otsu_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adapt_bin = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
        # Choose based on which has more reasonable connected components
        otsu_score = _count_char_like_components(cv2.bitwise_not(otsu_bin))
        adapt_score = _count_char_like_components(cv2.bitwise_not(adapt_bin))
        binary = otsu_bin if otsu_score >= adapt_score else adapt_bin
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean small noise with morphology
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    inverted = cv2.bitwise_not(binary)
    return binary, inverted


def _count_char_like_components(mask: np.ndarray) -> int:
    """Count components that look like characters."""
    h, w = mask.shape[:2]
    area = h * w
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    count = 0
    for i in range(1, num_labels):
        comp_area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Filter by size - relaxed thresholds
        if comp_area < area * 0.003 or comp_area > area * 0.30:
            continue
        # Filter by aspect ratio - more permissive
        ar = comp_w / float(comp_h) if comp_h > 0 else 0
        if not (0.1 <= ar <= 1.5):
            continue
        # Filter by height - lowered minimum
        if comp_h < h * 0.10 or comp_h > h * 0.95:
            continue
        count += 1
    
    return count


def _filter_components(mask: np.ndarray, min_area: float, max_area: float, ar_range: Tuple[float, float], min_height_ratio: float = 0.10) -> List[Tuple[int, int, int, int]]:
    """Filter connected components by size, aspect ratio, and height."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    h, w = mask.shape[:2]
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        ar = bw / float(bh or 1)
        if area < min_area or area > max_area:
            continue
        if not (ar_range[0] <= ar <= ar_range[1]):
            continue
        if bh < min_height_ratio * h:
            continue
        boxes.append((x, y, bw, bh))
    return boxes


def _sort_boxes_2line(boxes: Sequence[Tuple[int, int, int, int]], top_count: int = 3):
    """Sort boxes for 2-line plates: top row first (left-to-right), then bottom row.
    
    Uses gap-based separation instead of median, which works better when
    the number of characters differs between rows.
    """
    if not boxes:
        return []
    
    # Get center Y for each box
    cy = [(i, b[1] + b[3] / 2.0) for i, b in enumerate(boxes)]
    cy_sorted = sorted(cy, key=lambda x: x[1])  # sort by Y
    
    # Find the largest gap in Y positions to separate rows
    max_gap = 0
    split_idx = len(cy_sorted) // 2  # default to middle
    
    for i in range(len(cy_sorted) - 1):
        gap = cy_sorted[i+1][1] - cy_sorted[i][1]
        if gap > max_gap:
            max_gap = gap
            split_idx = i + 1
    
    # Need significant gap to split into rows (at least 20% of typical char height)
    heights = [b[3] for b in boxes]
    avg_height = np.mean(heights) if heights else 30
    min_gap_required = avg_height * 0.3
    
    if max_gap < min_gap_required:
        # No significant gap - treat as single line, sort by X
        return sorted(boxes, key=lambda b: b[0])
    
    # Split into top and bottom rows
    top_indices = set(cy_sorted[i][0] for i in range(split_idx))
    top = [boxes[i] for i in range(len(boxes)) if i in top_indices]
    bottom = [boxes[i] for i in range(len(boxes)) if i not in top_indices]
    
    # Sort each row by X position
    top = sorted(top, key=lambda b: b[0])
    bottom = sorted(bottom, key=lambda b: b[0])
    
    # Limit top row to expected count (usually 2-3 chars for Vietnamese plates)
    if len(top) > top_count:
        top = top[:top_count]
    
    return top + bottom


def normalize_characters(inverted_binary: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]], size: int = 28, pad: int = 2):
    chars: List[np.ndarray] = []
    for (x, y, w, h) in boxes:
        glyph = inverted_binary[y : y + h, x : x + w]
        side = max(w, h) + 2 * pad
        canvas = np.zeros((side, side), dtype=np.uint8)
        sx = (side - w) // 2
        sy = (side - h) // 2
        canvas[sy : sy + h, sx : sx + w] = glyph
        resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
        chars.append(resized)
    return chars


def segment_characters(
    plate_roi: np.ndarray, 
    plate_type: str = "car2",
    method: str = "auto",
    debug: bool = False,
) -> SegmentationResult:
    """
    Segment characters from a plate ROI.

    Args:
        plate_roi: Plate image (grayscale or BGR)
        plate_type: "car2" (square 2-line), "car1" (rect 1-line), "bike"
        method: Binarization method - "otsu", "adaptive", "combined", or "auto"
        debug: Include debug info in result
        
    Returns:
        SegmentationResult with character boxes and images
    """
    # Auto-select method based on plate type
    if method == "auto":
        # For corrected plates, combined method often works better
        method = "combined"
    
    binary, inverted = binarize_plate(plate_roi, method=method, denoise=True)
    h, w = inverted.shape[:2]
    area = h * w

    debug_info = {
        'plate_type': plate_type,
        'method': method,
        'roi_size': (w, h),
    }

    # Tuned ranges per plate type - RELAXED for better detection
    if plate_type == "car1":
        # Rectangular 1-line: characters are wider relative to plate
        min_area = area * 0.004
        max_area = area * 0.25
        ar_range = (0.15, 1.2)
        min_height_ratio = 0.12
        expected_rows = 1
        expected_chars = (7, 9)  # NNC-NNN.NN format
    elif plate_type == "bike":
        # Motorbike: 2 lines, smaller chars
        min_area = area * 0.003
        max_area = area * 0.25
        ar_range = (0.12, 1.3)
        min_height_ratio = 0.10
        expected_rows = 2
        expected_chars = (7, 9)
    else:  # car2 default (square 2-line)
        min_area = area * 0.004
        max_area = area * 0.25
        ar_range = (0.15, 1.2)
        min_height_ratio = 0.10
        expected_rows = 2
        expected_chars = (7, 9)

    mask = (inverted > 0).astype("uint8")
    boxes = _filter_components(mask, min_area=min_area, max_area=max_area, ar_range=ar_range, min_height_ratio=min_height_ratio)
    
    debug_info['raw_components'] = len(boxes)

    # Additional filtering: remove outliers by height AND area
    if boxes:
        heights = [b[3] for b in boxes]
        areas = [b[2] * b[3] for b in boxes]
        median_h = np.median(heights)
        median_area = np.median(areas)
        
        # Keep only boxes with height within 50% of median AND area within 60% of median
        # Also filter boxes at edge positions (x <= 3 or x+w >= w-3)
        margin = 3
        filtered_boxes = []
        for b in boxes:
            bx, by, bw, bh = b
            box_area = bw * bh
            
            # Height filter
            if not (0.5 * median_h <= bh <= 1.5 * median_h):
                continue
            
            # Area filter (detect thin artifacts)
            if box_area < 0.4 * median_area:
                continue
            
            # Edge filter (reject boxes touching ROI edge - likely plate border)
            if bx <= margin or bx + bw >= w - margin:
                continue
            
            filtered_boxes.append(b)
        
        boxes = filtered_boxes
    
    debug_info['filtered_components'] = len(boxes)

    # Sort by position
    if expected_rows == 2 and len(boxes) >= 4:
        boxes = _sort_boxes_2line(boxes, top_count=3)
    else:
        boxes = sorted(boxes, key=lambda b: b[0])

    # Limit to expected character count
    if len(boxes) > expected_chars[1]:
        # Keep the most central/best-sized characters
        boxes = _select_best_chars(boxes, expected_chars[1], h)
    
    debug_info['final_chars'] = len(boxes)

    chars = normalize_characters(inverted, boxes)
    return SegmentationResult(
        boxes=boxes, 
        char_images=chars, 
        inverted_binary=inverted, 
        binary=binary,
        debug_info=debug_info if debug else {},
    )


def _select_best_chars(
    boxes: List[Tuple[int, int, int, int]], 
    max_count: int,
    roi_height: int,
) -> List[Tuple[int, int, int, int]]:
    """Select best character candidates when too many are found."""
    if len(boxes) <= max_count:
        return boxes
    
    # Score each box by how "character-like" it is
    scored = []
    heights = [b[3] for b in boxes]
    median_h = np.median(heights)
    
    for box in boxes:
        x, y, w, h = box
        
        # Score based on:
        # 1. Height close to median
        h_score = 1.0 - abs(h - median_h) / median_h
        # 2. Aspect ratio close to 0.5 (typical for characters)
        ar = w / float(h) if h > 0 else 1
        ar_score = 1.0 - abs(ar - 0.5) / 0.5
        # 3. Not too close to edges
        edge_score = 1.0
        
        total_score = 0.4 * h_score + 0.4 * ar_score + 0.2 * edge_score
        scored.append((total_score, box))
    
    # Sort by score and keep top
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [s[1] for s in scored[:max_count]]
    
    # Re-sort by x position
    return sorted(selected, key=lambda b: b[0])


def segment_characters_multi_method(
    plate_roi: np.ndarray,
    plate_type: str = "car2",
) -> SegmentationResult:
    """
    Try multiple segmentation methods and return best result.
    
    Useful when single method fails to find enough characters.
    """
    methods = ["combined", "otsu", "adaptive"]
    best_result = None
    best_char_count = 0
    
    for method in methods:
        result = segment_characters(plate_roi, plate_type=plate_type, method=method)
        char_count = len(result.boxes)
        
        # Prefer results with 5-9 characters (typical for VN plates)
        if 5 <= char_count <= 9:
            return result
        
        if char_count > best_char_count:
            best_char_count = char_count
            best_result = result
    
    return best_result if best_result else SegmentationResult(
        boxes=[], char_images=[], inverted_binary=np.array([]), binary=np.array([])
    )
