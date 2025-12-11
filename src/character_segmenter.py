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
    aggressive: bool = False,
    light_preprocess: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binarize plate image for character segmentation.
    
    Args:
        plate_roi: Input plate ROI (grayscale or BGR)
        method: "otsu", "adaptive", or "combined"
        denoise: Apply denoising before binarization
        aggressive: If True, apply stronger morphology (may lose detail)
        light_preprocess: If True, use minimal preprocessing to preserve edge characters
        
    Returns:
        (binary, inverted_binary) tuple
    """
    gray = ensure_grayscale(plate_roi)
    
    if light_preprocess:
        # Minimal preprocessing - just light blur for noise reduction
        # This preserves characters at edges that may be lost with heavy preprocessing
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    else:
        # Standard preprocessing
        # Apply CLAHE for contrast enhancement - moderate clipLimit to preserve detail
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        
        # Optional light denoising - reduced strength to preserve character details
        if denoise:
            # Use lighter denoising to avoid losing thin strokes
            denoised = cv2.fastNlMeansDenoising(clahe, None, h=5, templateWindowSize=7, searchWindowSize=21)
        else:
            denoised = clahe
        
        # Light sharpening to enhance edges without creating artifacts
        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1,  5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        
        # Very light blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)

    if method == "adaptive":
        # Adaptive threshold - good for uneven lighting
        # Use smaller block size (15) for better character separation
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )
    elif method == "combined":
        # Try multiple methods and choose best one based on character-like regions
        _, otsu_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Try two adaptive variants for different plate conditions
        adapt_bin1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )
        adapt_bin2 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
        )
        # Choose based on which has more character-like connected components
        otsu_score = _count_char_like_components(cv2.bitwise_not(otsu_bin))
        adapt1_score = _count_char_like_components(cv2.bitwise_not(adapt_bin1))
        adapt2_score = _count_char_like_components(cv2.bitwise_not(adapt_bin2))
        
        # Pick the one with highest character count
        best_score = max(otsu_score, adapt1_score, adapt2_score)
        if adapt1_score == best_score:
            binary = adapt_bin1
        elif adapt2_score == best_score:
            binary = adapt_bin2
        else:
            binary = otsu_bin
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Light morphology - only if aggressive mode
    if aggressive:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    # Skip morphology when not aggressive to preserve edge characters

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


def _detect_two_line_layout(boxes: Sequence[Tuple[int, int, int, int]]) -> bool:
    """Detect if character boxes are arranged in 2 lines.
    
    Analyzes the Y-distribution of character centers to determine if there's
    a clear vertical gap indicating 2 rows of text.
    
    Args:
        boxes: List of (x, y, w, h) bounding boxes
        
    Returns:
        True if 2-line layout detected, False otherwise
    """
    if len(boxes) < 4:
        return False
    
    # Get center Y for each box
    centers_y = [b[1] + b[3] / 2.0 for b in boxes]
    centers_y_sorted = sorted(centers_y)
    
    # Calculate average character height
    heights = [b[3] for b in boxes]
    avg_height = np.mean(heights)
    
    # Find the largest gap between consecutive Y centers
    max_gap = 0
    for i in range(len(centers_y_sorted) - 1):
        gap = centers_y_sorted[i + 1] - centers_y_sorted[i]
        if gap > max_gap:
            max_gap = gap
    
    # If the largest gap is significant (> 30% of character height), it's 2 lines
    # This threshold works because characters in the same line have small Y variation
    # while the gap between lines is typically at least half the character height
    gap_threshold = avg_height * 0.3
    
    return max_gap > gap_threshold


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
    
    Vietnamese 2-line plates typically have:
    - Top row: 2-3 characters (province code + serial letter, e.g., "30G")
    - Bottom row: 4-5 characters (numbers, e.g., "51624")
    
    Algorithm:
    1. Find the vertical midpoint of the plate ROI
    2. Split boxes into top/bottom based on their center Y relative to midpoint
    3. Sort each row left-to-right by X position
    """
    if not boxes:
        return []
    
    if len(boxes) < 2:
        return list(boxes)
    
    # Calculate the vertical extent of all boxes
    all_tops = [b[1] for b in boxes]  # y coordinates
    all_bottoms = [b[1] + b[3] for b in boxes]  # y + h
    
    min_y = min(all_tops)
    max_y = max(all_bottoms)
    
    # The midpoint of the plate region
    mid_y = (min_y + max_y) / 2.0
    
    # Also compute average character height for validation
    heights = [b[3] for b in boxes]
    avg_height = np.mean(heights)
    
    # Split into top and bottom rows based on center Y
    top = []
    bottom = []
    
    for box in boxes:
        x, y, w, h = box
        center_y = y + h / 2.0
        
        # If center is above midpoint -> top row, else -> bottom row
        if center_y < mid_y:
            top.append(box)
        else:
            bottom.append(box)
    
    # Validate the split - if one row is empty, use gap-based fallback
    if not top or not bottom:
        # Fallback: find largest Y gap to split
        cy = [(i, b[1] + b[3] / 2.0) for i, b in enumerate(boxes)]
        cy_sorted = sorted(cy, key=lambda x: x[1])
        
        max_gap = 0
        split_idx = len(cy_sorted) // 2
        
        for i in range(len(cy_sorted) - 1):
            gap = cy_sorted[i+1][1] - cy_sorted[i][1]
            if gap > max_gap:
                max_gap = gap
                split_idx = i + 1
        
        # Check if gap is significant (at least 20% of char height)
        if max_gap < avg_height * 0.2:
            # No clear separation - treat as single line
            return sorted(boxes, key=lambda b: b[0])
        
        top_indices = set(cy_sorted[i][0] for i in range(split_idx))
        top = [boxes[i] for i in range(len(boxes)) if i in top_indices]
        bottom = [boxes[i] for i in range(len(boxes)) if i not in top_indices]
    
    # Sort each row by X position (left to right)
    top = sorted(top, key=lambda b: b[0])
    bottom = sorted(bottom, key=lambda b: b[0])
    
    # Vietnamese plates: top row usually has 2-3 chars, bottom has 4-5
    # Don't artificially limit - trust the segmentation
    
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
    
    # Try both preprocessing modes and pick the one that finds more characters
    # Heavy preprocessing can lose edge characters, light preprocessing may have more noise
    binary_heavy, inverted_heavy = binarize_plate(plate_roi, method=method, denoise=True, light_preprocess=False)
    binary_light, inverted_light = binarize_plate(plate_roi, method=method, denoise=False, light_preprocess=True)
    
    h, w = inverted_heavy.shape[:2]
    area = h * w

    debug_info = {
        'plate_type': plate_type,
        'method': method,
        'roi_size': (w, h),
    }

    # Tuned ranges per plate type - RELAXED for better detection
    if plate_type == "car1":
        # Rectangular 1-line: characters are wider relative to plate
        min_area = area * 0.003
        max_area = area * 0.25
        ar_range = (0.15, 1.5)  # Wider AR range for 1
        min_height_ratio = 0.15
        expected_rows = 1
        expected_chars = (7, 9)  # NNC-NNN.NN format
    elif plate_type == "bike":
        # Motorbike: 2 lines, smaller chars - MORE PERMISSIVE
        min_area = area * 0.002  # Smaller min for thin chars like 1
        max_area = area * 0.20
        ar_range = (0.10, 1.5)  # Allow thin chars (1, I) and wide chars
        min_height_ratio = 0.15  # Chars should be at least 15% of plate height
        expected_rows = 2
        expected_chars = (7, 10)  # VN plates: 7-9 chars typical
    else:  # car2 default (square 2-line)
        min_area = area * 0.003
        max_area = area * 0.25
        ar_range = (0.12, 1.5)
        min_height_ratio = 0.12
        expected_rows = 2
        expected_chars = (7, 10)

    # Helper function to extract boxes from a binary image
    def extract_boxes_from_binary(inverted_img):
        mask = (inverted_img > 0).astype("uint8")
        boxes = _filter_components(mask, min_area=min_area, max_area=max_area, ar_range=ar_range, min_height_ratio=min_height_ratio)
        
        if not boxes:
            return []
        
        # Additional filtering
        heights = [b[3] for b in boxes]
        areas = [b[2] * b[3] for b in boxes]
        median_h = np.median(heights)
        median_area = np.median(areas)
        
        margin = 2
        filtered_boxes = []
        for b in boxes:
            bx, by, bw, bh = b
            box_area = bw * bh
            
            if not (0.4 * median_h <= bh <= 1.6 * median_h):
                continue
            if box_area < 0.2 * median_area:
                continue
            # Slightly relaxed edge filter
            if bx <= margin or bx + bw >= w - margin:
                continue
            
            filtered_boxes.append(b)
        
        return filtered_boxes
    
    # Get boxes from both preprocessing modes
    boxes_heavy = extract_boxes_from_binary(inverted_heavy)
    boxes_light = extract_boxes_from_binary(inverted_light)
    
    # Choose the result with more characters (within expected range)
    # Light preprocessing often finds more edge characters
    if len(boxes_light) > len(boxes_heavy) and len(boxes_light) <= expected_chars[1]:
        boxes = boxes_light
        inverted = inverted_light
        binary = binary_light
        debug_info['preprocess_mode'] = 'light'
    else:
        boxes = boxes_heavy
        inverted = inverted_heavy
        binary = binary_heavy
        debug_info['preprocess_mode'] = 'heavy'
    
    debug_info['raw_components'] = len(boxes)
    debug_info['filtered_components'] = len(boxes)

    # Auto-detect if this is a 2-line plate based on Y distribution of boxes
    # This is more reliable than relying on plate_type parameter
    is_two_line = _detect_two_line_layout(boxes)
    
    # Sort by position
    if is_two_line and len(boxes) >= 4:
        boxes = _sort_boxes_2line(boxes, top_count=3)
        debug_info['detected_rows'] = 2
    else:
        boxes = sorted(boxes, key=lambda b: b[0])
        debug_info['detected_rows'] = 1

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
