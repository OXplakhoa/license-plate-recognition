from __future__ import annotations

"""
License plate detection helpers with configurable presets for car (2-line),
car rectangular (1-line), and motorbike plates.

Vietnamese License Plate Formats:
- Car square (2-line): ~1:1 to 1.5:1 aspect ratio, format NNC-NNN.NN
- Car rectangular (1-line): ~3:1 to 5:1 aspect ratio, format NNC-NNN.NN  
- Motorbike (2-line): ~1.5:1 to 2.5:1 aspect ratio, format NN-CN NNN.NN
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from .utils import ensure_grayscale


@dataclass
class DetectorConfig:
    """Configuration for license plate detection."""
    sigma: float = 0.33
    min_area_ratio: float = 0.02
    max_area_ratio: float = 0.25
    aspect_ratio_range: Tuple[float, float] = (0.8, 3.5)
    min_solidity: float = 0.5
    margin_ratio: float = 0.01  # Reduced from 0.03 to allow edge plates
    kernel_close: Tuple[int, int] | None = None
    kernel_close_ratio: Tuple[float, float] | None = None
    kernel_open: Tuple[int, int] | None = (5, 5)
    # Additional filters
    min_fill_ratio: float = 0.3  # Minimum ratio of contour area to bounding rect area
    use_bilateral_filter: bool = False  # Pre-filter to reduce noise while preserving edges


# ============================================================================
# PRESET CONFIGURATIONS FOR DIFFERENT PLATE TYPES
# ============================================================================
PRESETS: Dict[str, DetectorConfig] = {
    # Biển xe hơi vuông 2 dòng (Square car plate - 2 lines)
    # Tỷ lệ W:H khoảng 1:1 đến 1.5:1
    "car_square": DetectorConfig(
        min_area_ratio=0.005,  # Reduced from 0.008 to catch smaller plates
        max_area_ratio=0.30,
        aspect_ratio_range=(0.7, 2.2),  # Wider range for tilted plates
        min_solidity=0.40,  # Reduced from 0.45
        margin_ratio=0.005,  # Very small margin
        kernel_close=(25, 7),
        kernel_open=(5, 5),
        min_fill_ratio=0.30,
    ),
    
    # Biển xe hơi chữ nhật 1 dòng (Rectangular car plate - 1 line)
    # Tỷ lệ W:H khoảng 3:1 đến 5:1
    "car_rect": DetectorConfig(
        min_area_ratio=0.002,  # Reduced to catch very small rectangular plates
        max_area_ratio=0.20,
        aspect_ratio_range=(2.5, 7.0),  # Wider aspect range
        min_solidity=0.35,
        margin_ratio=0.005,
        kernel_close=(45, 5),  # Wider kernel for long plates
        kernel_open=(7, 3),
        min_fill_ratio=0.25,
    ),
    
    # Biển xe máy 2 dòng (Motorbike plate - 2 lines)
    # Tỷ lệ W:H khoảng 1.5:1 đến 2.5:1
    "bike": DetectorConfig(
        min_area_ratio=0.003,  # Reduced from 0.005
        max_area_ratio=0.15,
        aspect_ratio_range=(1.0, 3.2),  # Wider range
        min_solidity=0.35,
        margin_ratio=0.005,
        kernel_close=(20, 7),
        kernel_open=(5, 3),
        min_fill_ratio=0.25,
    ),
    
    # Legacy aliases for backward compatibility
    "car2": DetectorConfig(
        min_area_ratio=0.008,
        max_area_ratio=0.30,
        aspect_ratio_range=(0.7, 2.2),
        min_solidity=0.40,
        margin_ratio=0.005,
        kernel_close=(25, 7),
        kernel_open=(5, 5),
        min_fill_ratio=0.30,
    ),
    "car1": DetectorConfig(
        min_area_ratio=0.005,
        max_area_ratio=0.20,
        aspect_ratio_range=(2.5, 7.0),
        min_solidity=0.35,
        margin_ratio=0.005,
        kernel_close=(45, 5),
        kernel_open=(7, 3),
        min_fill_ratio=0.25,
    ),
}

# Default presets to use for multi-preset detection
DEFAULT_MULTI_PRESETS = ["car_square", "car_rect", "bike"]


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    v = float(np.median(image))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper


def _kernel_from_ratio(shape: Tuple[int, int], ratio: Tuple[float, float]) -> Tuple[int, int]:
    h, w = shape
    kw = max(5, int(w * ratio[0]))
    kh = max(3, int(h * ratio[1]))
    if kw % 2 == 0:
        kw += 1
    if kh % 2 == 0:
        kh += 1
    return kw, kh


def detect_license_plates(gray_image: np.ndarray, mode: str = "car2", config: DetectorConfig | None = None, debug: bool = False):
    """Detect candidate license plate bounding boxes on a preprocessed grayscale image.

    Returns (plates, debug_info) where plates is a list of (x, y, w, h).
    """
    cfg = config or PRESETS.get(mode, PRESETS["car2"])
    gray = ensure_grayscale(gray_image)
    h, w = gray.shape[:2]
    img_area = h * w

    lower, upper = auto_canny(gray, sigma=cfg.sigma)
    edges = cv2.Canny(gray, lower, upper)

    if cfg.kernel_close_ratio is not None:
        close_size = _kernel_from_ratio((h, w), cfg.kernel_close_ratio)
    else:
        close_size = cfg.kernel_close or (25, 5)
    open_size = cfg.kernel_open or (5, 5)

    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, close_size))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, open_size))

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plates: List[Tuple[int, int, int, int]] = []
    rejected = {"area": 0, "aspect": 0, "solidity": 0, "margin": 0}
    margin_px = max(3, int(w * cfg.margin_ratio))

    min_area = img_area * cfg.min_area_ratio
    max_area = img_area * cfg.max_area_ratio

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        aspect = bw / float(bh or 1)

        if not (min_area <= area <= max_area):
            rejected["area"] += 1
            continue
        if not (cfg.aspect_ratio_range[0] <= aspect <= cfg.aspect_ratio_range[1]):
            rejected["aspect"] += 1
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_area
        if solidity < cfg.min_solidity:
            rejected["solidity"] += 1
            continue

        if x < margin_px or y < margin_px or (x + bw) > (w - margin_px) or (y + bh) > (h - margin_px):
            rejected["margin"] += 1
            continue

        plates.append((x, y, bw, bh))

    debug_info = {
        "edges": edges,
        "morph": morph,
        "contours": len(contours),
        "canny_lower": lower,
        "canny_upper": upper,
        "kernel_close": close_size,
        "kernel_open": open_size,
        "rejected": rejected,
    }
    if debug:
        debug_info["plates"] = plates
    return plates, debug_info


def detect_with_enhanced_preprocessing(
    gray_image: np.ndarray, 
    mode: str = "car_square", 
    config: DetectorConfig | None = None,
    debug: bool = False
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Detect plates using enhanced preprocessing with CLAHE and morphological gradient.
    
    This method is better for images with low contrast or uneven lighting.
    """
    cfg = config or PRESETS.get(mode, PRESETS["car_square"])
    gray = ensure_grayscale(gray_image)
    h, w = gray.shape[:2]
    img_area = h * w
    
    # Step 1: CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Step 2: Bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Step 3: Morphological gradient to highlight edges
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel_grad)
    
    # Step 4: Threshold to get binary
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 5: Close gaps to form connected regions (like plates)
    close_size = cfg.kernel_close or (25, 7)
    open_size = cfg.kernel_open or (5, 5)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, 
                             cv2.getStructuringElement(cv2.MORPH_RECT, close_size))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, 
                             cv2.getStructuringElement(cv2.MORPH_RECT, open_size))
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plates: List[Tuple[int, int, int, int]] = []
    rejected = {"area": 0, "aspect": 0, "solidity": 0, "margin": 0}
    margin_px = max(2, int(w * cfg.margin_ratio))
    
    min_area = img_area * cfg.min_area_ratio
    max_area = img_area * cfg.max_area_ratio
    
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        aspect = bw / float(bh or 1)
        
        if not (min_area <= area <= max_area):
            rejected["area"] += 1
            continue
        if not (cfg.aspect_ratio_range[0] <= aspect <= cfg.aspect_ratio_range[1]):
            rejected["aspect"] += 1
            continue
        
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_area
        if solidity < cfg.min_solidity:
            rejected["solidity"] += 1
            continue
        
        # Relaxed margin check - allow touching edges
        if x < margin_px or y < margin_px or (x + bw) > (w - margin_px) or (y + bh) > (h - margin_px):
            rejected["margin"] += 1
            continue
        
        plates.append((x, y, bw, bh))
    
    debug_info = {
        "enhanced": enhanced,
        "gradient": gradient,
        "binary": binary,
        "morph": morph,
        "contours": len(contours),
        "rejected": rejected,
        "method": "enhanced_preprocessing"
    }
    
    return plates, debug_info


def detect_with_tophat(
    gray_image: np.ndarray,
    mode: str = "car_square",
    config: DetectorConfig | None = None,
    debug: bool = False
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Detect plates using tophat transform - good for bright plates on dark background."""
    cfg = config or PRESETS.get(mode, PRESETS["car_square"])
    gray = ensure_grayscale(gray_image)
    h, w = gray.shape[:2]
    img_area = h * w
    
    # Black tophat to find dark regions (or white tophat for bright plates)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_tophat)
    
    # Combine both
    combined = cv2.add(tophat, blackhat)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)
    
    # Threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphology
    close_size = cfg.kernel_close or (25, 7)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, close_size))
    
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plates: List[Tuple[int, int, int, int]] = []
    rejected = {"area": 0, "aspect": 0, "solidity": 0, "margin": 0}
    margin_px = max(2, int(w * cfg.margin_ratio))
    min_area = img_area * cfg.min_area_ratio
    max_area = img_area * cfg.max_area_ratio
    
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        aspect = bw / float(bh or 1)
        
        if not (min_area <= area <= max_area):
            rejected["area"] += 1
            continue
        if not (cfg.aspect_ratio_range[0] <= aspect <= cfg.aspect_ratio_range[1]):
            rejected["aspect"] += 1
            continue
        
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_area
        if solidity < cfg.min_solidity:
            rejected["solidity"] += 1
            continue
        
        if x < margin_px or y < margin_px or (x + bw) > (w - margin_px) or (y + bh) > (h - margin_px):
            rejected["margin"] += 1
            continue
        
        plates.append((x, y, bw, bh))
    
    debug_info = {
        "tophat": tophat,
        "blackhat": blackhat,
        "enhanced": enhanced,
        "binary": binary,
        "morph": morph,
        "contours": len(contours),
        "rejected": rejected,
        "method": "tophat"
    }
    
    return plates, debug_info


# ============================================================================
# NON-MAXIMUM SUPPRESSION (NMS) FOR MERGING OVERLAPPING BOXES
# ============================================================================

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Tuples of (x, y, w, h)
    
    Returns:
        IoU score between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Calculate intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height
    
    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(
    boxes: List[Tuple[int, int, int, int]],
    scores: Optional[List[float]] = None,
    iou_threshold: float = 0.3
) -> List[Tuple[int, int, int, int]]:
    """Apply Non-Maximum Suppression to remove overlapping bounding boxes.
    
    Args:
        boxes: List of (x, y, w, h) bounding boxes
        scores: Optional confidence scores for each box. If None, uses box area as score.
        iou_threshold: IoU threshold for suppression (boxes with IoU > threshold are merged)
    
    Returns:
        List of filtered bounding boxes after NMS
    """
    if not boxes:
        return []
    
    # If no scores provided, use area as score (larger boxes preferred)
    if scores is None:
        scores = [w * h for (x, y, w, h) in boxes]
    
    # Convert to numpy for easier manipulation
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    
    # Sort by score (descending)
    indices = np.argsort(scores_np)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Pick the box with highest score
        current_idx = indices[0]
        keep.append(current_idx)
        
        if len(indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        remaining_indices = indices[1:]
        ious = np.array([
            compute_iou(tuple(boxes_np[current_idx]), tuple(boxes_np[idx]))
            for idx in remaining_indices
        ])
        
        # Keep boxes with IoU below threshold
        mask = ious < iou_threshold
        indices = remaining_indices[mask]
    
    return [tuple(boxes_np[i]) for i in keep]


def merge_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.2
) -> List[Tuple[int, int, int, int]]:
    """Merge significantly overlapping boxes into their union.
    
    Unlike NMS which keeps only one box, this merges overlapping boxes
    into a single larger box that encompasses both.
    
    Args:
        boxes: List of (x, y, w, h) bounding boxes
        iou_threshold: Minimum IoU to trigger merging
    
    Returns:
        List of merged bounding boxes
    """
    if not boxes:
        return []
    
    boxes = list(boxes)
    merged = True
    
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
            
            current = boxes[i]
            x1, y1, w1, h1 = current
            
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                
                iou = compute_iou(current, boxes[j])
                if iou > iou_threshold:
                    # Merge boxes
                    x2, y2, w2, h2 = boxes[j]
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    current = (new_x, new_y, new_w, new_h)
                    x1, y1, w1, h1 = current
                    used[j] = True
                    merged = True
            
            new_boxes.append(current)
            used[i] = True
        
        boxes = new_boxes
    
    return boxes


# ============================================================================
# MULTI-PRESET DETECTION
# ============================================================================

def detect_multi_preset(
    gray_image: np.ndarray,
    presets: Optional[List[str]] = None,
    merge_method: str = "nms",
    iou_threshold: float = 0.3,
    debug: bool = False
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Detect license plates using multiple presets and merge results.
    
    This function runs detection with multiple preset configurations
    (car_square, car_rect, bike) and combines the results using NMS
    to avoid duplicate detections.
    
    Args:
        gray_image: Grayscale preprocessed image
        presets: List of preset names to use. Defaults to DEFAULT_MULTI_PRESETS
        merge_method: "nms" for non-maximum suppression, "merge" for union merge
        iou_threshold: IoU threshold for merging/suppressing
        debug: Return debug information
    
    Returns:
        (plates, debug_info) where plates is list of (x, y, w, h) and
        debug_info contains per-preset results
    """
    if presets is None:
        presets = DEFAULT_MULTI_PRESETS
    
    all_boxes = []
    all_scores = []
    preset_results = {}
    
    # Method 1: Standard Canny edge detection
    for preset_name in presets:
        if preset_name not in PRESETS:
            continue
        
        plates, info = detect_license_plates(gray_image, mode=preset_name, debug=debug)
        
        for (x, y, w, h) in plates:
            area = w * h
            all_boxes.append((x, y, w, h))
            all_scores.append(area)
        
        preset_results[f"canny_{preset_name}"] = {
            "plates": plates,
            "count": len(plates),
        }
    
    # Method 2: Enhanced preprocessing (CLAHE + morph gradient)
    for preset_name in presets:
        if preset_name not in PRESETS:
            continue
        
        plates, info = detect_with_enhanced_preprocessing(gray_image, mode=preset_name, debug=debug)
        
        for (x, y, w, h) in plates:
            area = w * h
            all_boxes.append((x, y, w, h))
            all_scores.append(area)
        
        preset_results[f"enhanced_{preset_name}"] = {
            "plates": plates,
            "count": len(plates),
        }
    
    # Method 3: Tophat transform  
    for preset_name in presets:
        if preset_name not in PRESETS:
            continue
        
        plates, info = detect_with_tophat(gray_image, mode=preset_name, debug=debug)
        
        for (x, y, w, h) in plates:
            area = w * h
            all_boxes.append((x, y, w, h))
            all_scores.append(area)
        
        preset_results[f"tophat_{preset_name}"] = {
            "plates": plates,
            "count": len(plates),
        }
    
    # Merge overlapping detections
    if merge_method == "nms":
        final_plates = non_maximum_suppression(all_boxes, all_scores, iou_threshold)
    else:
        final_plates = merge_overlapping_boxes(all_boxes, iou_threshold)
    
    debug_info = {
        "preset_results": preset_results,
        "total_candidates": len(all_boxes),
        "final_count": len(final_plates),
        "merge_method": merge_method,
        "presets_used": presets,
        "methods_used": ["canny", "enhanced", "tophat"],
    }
    
    return final_plates, debug_info


def detect_with_fallback(
    gray_image: np.ndarray,
    primary_presets: Optional[List[str]] = None,
    fallback_presets: Optional[List[str]] = None,
    min_plates: int = 1,
    debug: bool = False
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Detect plates with fallback to alternative presets if primary fails.
    
    Args:
        gray_image: Grayscale preprocessed image
        primary_presets: Primary presets to try first
        fallback_presets: Fallback presets if primary finds fewer than min_plates
        min_plates: Minimum plates required to avoid fallback
        debug: Return debug information
    
    Returns:
        (plates, debug_info)
    """
    if primary_presets is None:
        primary_presets = ["car_square"]
    if fallback_presets is None:
        fallback_presets = ["car_rect", "bike"]
    
    # Try primary presets first
    plates, info = detect_multi_preset(gray_image, primary_presets, debug=debug)
    
    if len(plates) >= min_plates:
        info["fallback_used"] = False
        return plates, info
    
    # Fallback to additional presets
    all_presets = primary_presets + fallback_presets
    plates, info = detect_multi_preset(gray_image, all_presets, debug=debug)
    info["fallback_used"] = True
    
    return plates, info


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_plate_type_from_aspect(aspect_ratio: float, width: int = 0, height: int = 0) -> str:
    """Classify plate type based on aspect ratio and size.
    
    Vietnamese plate sizes (actual):
    - Car square (2-line): ~190x110mm → AR ≈ 1.73
    - Car rect (1-line): ~470x110mm → AR ≈ 4.27  
    - Bike (2-line): ~140x90mm → AR ≈ 1.56
    
    The challenge: car_square and bike have similar AR (~1.5-2.0)
    Use size as secondary criterion: car plates are larger than bike plates
    
    Args:
        aspect_ratio: Width/Height ratio of the detected plate
        width: Width of detected ROI in pixels (optional, for size-based distinction)
        height: Height of detected ROI in pixels (optional)
    
    Returns:
        Plate type string: "car2" (square 2-line), "car1" (rect 1-line), or "bike"
    """
    # Long rectangular plate - definitely car 1-line
    if aspect_ratio > 2.5:
        return "car1"
    
    # Very square plate (AR < 1.4) - likely car 2-line  
    if aspect_ratio < 1.4:
        return "car2"
    
    # Medium AR (1.4 - 2.5): Could be car2 or bike
    # Use size to distinguish: car plates are typically larger
    if width > 0 and height > 0:
        area = width * height
        # Car plates are usually larger (wider detection in image)
        # Typical thresholds: car ROI > 10000 pixels, bike < 8000 pixels
        # Also car plates tend to be wider absolutely
        if width > 130 or area > 9000:
            return "car2"  # Larger = car
        elif width < 100 and area < 6000:
            return "bike"  # Smaller = bike
    
    # Default fallback based on AR alone
    # AR 1.4-1.8: more likely car2 (square plates tend to be detected with AR ~1.5-1.8)
    # AR 1.8-2.5: could be either, but bike plates when detected tend to have AR ~1.5-1.7
    if aspect_ratio < 1.8:
        return "car2"
    else:
        return "bike"


def classify_detected_plates(
    plates: List[Tuple[int, int, int, int]]
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """Classify detected plates by their aspect ratio.
    
    Args:
        plates: List of (x, y, w, h) bounding boxes
    
    Returns:
        Dictionary mapping plate type to list of plates
    """
    classified = {"car2": [], "car1": [], "bike": []}
    
    for (x, y, w, h) in plates:
        aspect = w / float(h or 1)
        plate_type = get_plate_type_from_aspect(aspect, w, h)
        classified[plate_type].append((x, y, w, h))
    
    return classified


# ============================================================================
# PHASE 2: PERSPECTIVE CORRECTION
# ============================================================================

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    Returns:
        Ordered points array
    """
    # Sort by sum (top-left has smallest sum, bottom-right has largest)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # Sort by difference (top-right has smallest diff, bottom-left has largest)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def find_plate_contour(
    gray_image: np.ndarray,
    roi: Tuple[int, int, int, int],
    expand_ratio: float = 0.1
) -> Optional[np.ndarray]:
    """Find the 4-corner contour of a license plate within an ROI.
    
    Args:
        gray_image: Grayscale image
        roi: Bounding box (x, y, w, h) of detected plate region
        expand_ratio: How much to expand ROI for better contour detection
    
    Returns:
        4 corner points of the plate contour, or None if not found
    """
    x, y, w, h = roi
    img_h, img_w = gray_image.shape[:2]
    
    # Expand ROI slightly
    expand_x = int(w * expand_ratio)
    expand_y = int(h * expand_ratio)
    x1 = max(0, x - expand_x)
    y1 = max(0, y - expand_y)
    x2 = min(img_w, x + w + expand_x)
    y2 = min(img_h, y + h + expand_y)
    
    # Extract ROI
    roi_img = gray_image[y1:y2, x1:x2]
    
    if roi_img.size == 0:
        return None
    
    # Edge detection
    blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour that can be approximated to 4 points
    best_approx = None
    best_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Skip tiny contours
            continue
        
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # We want exactly 4 corners
        if len(approx) == 4 and area > best_area:
            best_approx = approx
            best_area = area
    
    if best_approx is None:
        # Fallback: use minimum area rectangle
        all_points = np.vstack(contours)
        rect = cv2.minAreaRect(all_points)
        box = cv2.boxPoints(rect)
        best_approx = np.int32(box).reshape(-1, 1, 2)
    
    # Convert to absolute coordinates
    corners = best_approx.reshape(4, 2).astype("float32")
    corners[:, 0] += x1
    corners[:, 1] += y1
    
    return order_points(corners)


def perspective_transform(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Apply perspective transformation to extract and rectify a plate region.
    
    Args:
        image: Input image (grayscale or BGR)
        corners: 4 corner points in order [top-left, top-right, bottom-right, bottom-left]
        output_size: (width, height) of output. If None, auto-calculate from corners.
    
    Returns:
        Rectified (warped) plate image
    """
    tl, tr, br, bl = corners
    
    # Calculate output dimensions if not specified
    if output_size is None:
        # Width: max of top and bottom edge lengths
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        width = int(max(width_top, width_bottom))
        
        # Height: max of left and right edge lengths
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        height = int(max(height_left, height_right))
        
        # Ensure minimum dimensions
        width = max(width, 100)
        height = max(height, 30)
    else:
        width, height = output_size
    
    # Destination points (rectangle)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst)
    
    # Apply warp
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped


def detect_and_correct_plate(
    gray_image: np.ndarray,
    roi: Tuple[int, int, int, int],
    debug: bool = False
) -> Tuple[Optional[np.ndarray], Dict]:
    """Detect plate corners within ROI and apply perspective correction.
    
    Args:
        gray_image: Grayscale preprocessed image
        roi: Bounding box (x, y, w, h) from initial detection
        debug: Return debug information
    
    Returns:
        (corrected_plate, debug_info) where corrected_plate is the rectified image
    """
    x, y, w, h = roi
    
    debug_info = {
        "original_roi": roi,
        "corners": None,
        "success": False,
    }
    
    # Find 4-corner contour
    corners = find_plate_contour(gray_image, roi)
    
    if corners is None:
        # Fallback: just crop the bounding box
        plate_crop = gray_image[y:y+h, x:x+w]
        debug_info["method"] = "crop_only"
        return plate_crop, debug_info
    
    debug_info["corners"] = corners
    
    # Apply perspective transform
    corrected = perspective_transform(gray_image, corners)
    debug_info["success"] = True
    debug_info["method"] = "perspective"
    debug_info["output_size"] = corrected.shape[::-1] if len(corrected.shape) == 2 else corrected.shape[1::-1]
    
    return corrected, debug_info


def detect_plates_with_correction(
    gray_image: np.ndarray,
    presets: Optional[List[str]] = None,
    apply_correction: bool = True,
    debug: bool = False
) -> Tuple[List[Tuple[np.ndarray, Tuple[int, int, int, int]]], Dict]:
    """Full pipeline: detect plates, find corners, apply perspective correction.
    
    Args:
        gray_image: Grayscale preprocessed image
        presets: Presets to use for detection
        apply_correction: Whether to apply perspective correction
        debug: Return debug information
    
    Returns:
        (plates, debug_info) where plates is list of (corrected_image, original_roi)
    """
    # Step 1: Detect plates using multi-preset
    rois, detect_info = detect_multi_preset(gray_image, presets, debug=debug)
    
    results = []
    correction_info = []
    
    for roi in rois:
        if apply_correction:
            # Step 2: Find corners and correct perspective
            corrected, corr_info = detect_and_correct_plate(gray_image, roi, debug=debug)
            results.append((corrected, roi))
            correction_info.append(corr_info)
        else:
            # Just crop
            x, y, w, h = roi
            crop = gray_image[y:y+h, x:x+w]
            results.append((crop, roi))
            correction_info.append({"method": "crop_only", "success": True})
    
    debug_info = {
        "detection": detect_info,
        "correction": correction_info,
        "total_plates": len(results),
    }
    
    return results, debug_info


def compute_skew_angle(image: np.ndarray) -> float:
    """Compute the skew angle of text/plate in an image using Hough lines.
    
    Args:
        image: Grayscale image
    
    Returns:
        Skew angle in degrees (positive = clockwise rotation needed)
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Calculate angles of all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # Only consider near-horizontal lines
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Return median angle
    return float(np.median(angles))


def deskew_image(image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """Deskew (rotate) an image to correct for tilted text.
    
    Args:
        image: Input image (grayscale or BGR)
        angle: Rotation angle in degrees. If None, auto-detect.
    
    Returns:
        Deskewed image
    """
    if angle is None:
        angle = compute_skew_angle(image)
    
    if abs(angle) < 0.5:  # Skip if nearly straight
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust rotation matrix for new center
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Apply rotation
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def correct_plate_perspective_and_skew(
    gray_image: np.ndarray,
    roi: Tuple[int, int, int, int],
    deskew: bool = True,
    debug: bool = False
) -> Tuple[np.ndarray, Dict]:
    """Complete correction pipeline: perspective + deskew.
    
    Args:
        gray_image: Grayscale image
        roi: Detected plate ROI (x, y, w, h)
        deskew: Also apply deskew after perspective correction
        debug: Return debug info
    
    Returns:
        (corrected_image, debug_info)
    """
    # Step 1: Perspective correction
    corrected, debug_info = detect_and_correct_plate(gray_image, roi, debug=debug)
    
    if corrected is None or corrected.size == 0:
        x, y, w, h = roi
        corrected = gray_image[y:y+h, x:x+w]
    
    # Step 2: Deskew
    if deskew and corrected is not None and corrected.size > 0:
        skew_angle = compute_skew_angle(corrected)
        debug_info["skew_angle"] = skew_angle
        
        if abs(skew_angle) > 0.5:
            corrected = deskew_image(corrected, skew_angle)
            debug_info["deskew_applied"] = True
        else:
            debug_info["deskew_applied"] = False
    
    return corrected, debug_info


# ============================================================================
# PHASE 3: EDGE DENSITY BACKUP DETECTOR
# ============================================================================

def compute_edge_density(
    image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None
) -> float:
    """Compute edge density (ratio of edge pixels) in an image or ROI.
    
    License plates typically have high edge density due to characters.
    This can be used to validate candidate regions or as a scoring metric.
    
    Args:
        image: Grayscale image
        roi: Optional (x, y, w, h) region. If None, computes for entire image.
    
    Returns:
        Edge density as ratio (0 to 1)
    """
    gray = ensure_grayscale(image)
    
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]
    
    if gray.size == 0:
        return 0.0
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Calculate edge density
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = np.count_nonzero(edges)
    
    return edge_pixels / total_pixels if total_pixels > 0 else 0.0


def compute_vertical_edge_score(image: np.ndarray) -> float:
    """Compute score based on vertical edges (common in characters).
    
    License plates have many vertical edges from characters like
    1, I, L, T, etc. This helps distinguish plates from other regions.
    
    Args:
        image: Grayscale image of candidate region
    
    Returns:
        Vertical edge score (higher = more vertical edges)
    """
    gray = ensure_grayscale(image)
    
    if gray.size == 0:
        return 0.0
    
    # Sobel filter for vertical edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude of vertical vs horizontal edges
    vert_mag = np.abs(sobel_x).mean()
    horiz_mag = np.abs(sobel_y).mean()
    
    # Ratio favoring vertical edges
    total = vert_mag + horiz_mag
    if total < 1e-6:
        return 0.0
    
    return vert_mag / total


def compute_contrast_score(image: np.ndarray) -> float:
    """Compute contrast score for candidate region.
    
    License plates typically have high contrast (light background, dark text
    or vice versa).
    
    Args:
        image: Grayscale image of candidate region
    
    Returns:
        Contrast score (0 to 1, higher = more contrast)
    """
    gray = ensure_grayscale(image)
    
    if gray.size == 0:
        return 0.0
    
    # Standard deviation as contrast measure
    std = np.std(gray)
    # Normalize to 0-1 range (assuming max std ~80 for good plates)
    return min(1.0, std / 80.0)


def compute_plate_score(
    image: np.ndarray,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """Compute combined score for license plate likelihood.
    
    Combines edge density, vertical edges, and contrast into a single score.
    
    Args:
        image: Grayscale image of candidate region
        weights: Optional weights for each metric
    
    Returns:
        (total_score, individual_scores_dict)
    """
    if weights is None:
        weights = {
            "edge_density": 0.4,
            "vertical_edges": 0.3,
            "contrast": 0.3
        }
    
    scores = {
        "edge_density": compute_edge_density(image),
        "vertical_edges": compute_vertical_edge_score(image),
        "contrast": compute_contrast_score(image)
    }
    
    total = sum(weights.get(k, 0) * v for k, v in scores.items())
    
    return total, scores


def sliding_window_detect(
    gray_image: np.ndarray,
    window_sizes: Optional[List[Tuple[int, int]]] = None,
    step_ratio: float = 0.3,
    score_threshold: float = 0.45,
    aspect_ratio_range: Tuple[float, float] = (0.7, 5.0),
    nms_threshold: float = 0.2,
    max_candidates: int = 10,
    debug: bool = False
) -> Tuple[List[Dict], Dict]:
    """Detect license plates using sliding window and edge density scoring.
    
    This is a backup detector for when contour-based detection fails.
    It slides windows of various sizes across the image and scores each
    region based on edge density and other features.
    
    Args:
        gray_image: Grayscale image
        window_sizes: List of (width, height) window sizes. If None, auto-generate.
        step_ratio: Step size as ratio of window size (0.3 = 30% overlap)
        score_threshold: Minimum score to consider as candidate (default 0.45)
        aspect_ratio_range: Valid aspect ratio range for plates
        nms_threshold: IoU threshold for non-maximum suppression
        max_candidates: Maximum number of candidates to return
        debug: Return debug information
    
    Returns:
        (detections, debug_info) where detections is list of dicts with
        'box', 'score', 'scores' keys
    """
    gray = ensure_grayscale(gray_image)
    h, w = gray.shape[:2]
    
    # Auto-generate window sizes based on image size
    if window_sizes is None:
        # Typical plate sizes as fraction of image
        window_sizes = []
        for scale in [0.15, 0.2, 0.25, 0.3]:
            for aspect in [1.0, 1.5, 2.5, 4.0]:
                win_h = int(h * scale)
                win_w = int(win_h * aspect)
                if win_w < w and win_h < h:
                    window_sizes.append((win_w, win_h))
    
    candidates = []
    windows_checked = 0
    
    for win_w, win_h in window_sizes:
        step_x = max(1, int(win_w * step_ratio))
        step_y = max(1, int(win_h * step_ratio))
        
        for y in range(0, h - win_h, step_y):
            for x in range(0, w - win_w, step_x):
                windows_checked += 1
                
                # Extract window
                window = gray[y:y+win_h, x:x+win_w]
                
                # Check aspect ratio
                aspect = win_w / float(win_h)
                if not (aspect_ratio_range[0] <= aspect <= aspect_ratio_range[1]):
                    continue
                
                # Compute score
                score, scores = compute_plate_score(window)
                
                if score >= score_threshold:
                    candidates.append({
                        "box": (x, y, win_w, win_h),
                        "score": score,
                        "scores": scores,
                        "aspect_ratio": aspect
                    })
    
    # Apply NMS to reduce overlapping detections
    if candidates:
        boxes = [c["box"] for c in candidates]
        scores_list = [c["score"] for c in candidates]
        keep_boxes = non_maximum_suppression(boxes, scores_list, nms_threshold)
        
        # Filter candidates to keep only NMS survivors
        keep_set = set(keep_boxes)
        candidates = [c for c in candidates if c["box"] in keep_set]
    
    # Sort by score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    
    # Limit to max_candidates
    candidates = candidates[:max_candidates]
    
    debug_info = {
        "windows_checked": windows_checked,
        "candidates_before_nms": len(candidates),
        "window_sizes": window_sizes,
        "threshold": score_threshold
    }
    
    return candidates, debug_info


def detect_with_edge_backup(
    gray_image: np.ndarray,
    presets: Optional[List[str]] = None,
    min_plates: int = 1,
    edge_score_threshold: float = 0.45,
    max_edge_candidates: int = 5,
    debug: bool = False
) -> Tuple[List[Dict], Dict]:
    """Detect plates with fallback to edge density detector.
    
    First tries contour-based multi-preset detection. If that fails to
    find enough plates, falls back to sliding window edge density detection.
    
    Args:
        gray_image: Grayscale image
        presets: Presets for primary detection
        min_plates: Minimum plates to find before using backup
        edge_score_threshold: Score threshold for edge density detector
        max_edge_candidates: Maximum candidates from edge detector
        debug: Return debug info
    
    Returns:
        (detections, debug_info) where detections have 'box', 'method', etc.
    """
    gray = ensure_grayscale(gray_image)
    
    # Try primary detection first
    primary_plates, primary_info = detect_multi_preset(gray, presets=presets, debug=debug)
    
    results = []
    for plate in primary_plates:
        x, y, w, h = plate
        results.append({
            "box": plate,
            "method": "contour",
            "score": 1.0,  # Contour detection is high confidence
            "aspect_ratio": w / float(h)
        })
    
    debug_info = {
        "primary_method": "contour",
        "primary_count": len(primary_plates),
        "primary_info": primary_info if debug else None,
        "backup_used": False
    }
    
    # If not enough plates found, use backup
    if len(results) < min_plates:
        backup_plates, backup_info = sliding_window_detect(
            gray,
            score_threshold=edge_score_threshold,
            max_candidates=max_edge_candidates,
            debug=debug
        )
        
        # Add backup detections (avoiding duplicates)
        for bp in backup_plates:
            # Check if this overlaps significantly with existing detections
            is_duplicate = False
            for existing in results:
                iou = compute_iou(bp["box"], existing["box"])
                if iou > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                bp["method"] = "edge_density"
                results.append(bp)
        
        debug_info["backup_used"] = True
        debug_info["backup_count"] = len(backup_plates)
        debug_info["backup_info"] = backup_info if debug else None
    
    return results, debug_info


# ============================================================================
# PHASE 4: CHARACTER-BASED ROI VALIDATION
# ============================================================================

def detect_character_candidates(
    roi_image: np.ndarray,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.25,
    aspect_ratio_range: Tuple[float, float] = (0.1, 1.5),
    min_height_ratio: float = 0.15,
    max_height_ratio: float = 0.9
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Detect character-like components in a plate ROI.
    
    This function finds connected components that look like characters
    based on size, aspect ratio, and position constraints.
    
    Args:
        roi_image: Grayscale or BGR image of the plate region
        min_area_ratio: Minimum component area as ratio of ROI area
        max_area_ratio: Maximum component area as ratio of ROI area
        aspect_ratio_range: Valid aspect ratio (width/height) for characters
        min_height_ratio: Minimum character height as ratio of ROI height
        max_height_ratio: Maximum character height as ratio of ROI height
    
    Returns:
        (character_boxes, debug_info) where boxes are (x, y, w, h)
    """
    gray = ensure_grayscale(roi_image)
    h, w = gray.shape[:2]
    roi_area = h * w
    
    if roi_area == 0:
        return [], {"error": "Empty ROI"}
    
    # Relax parameters for small ROI (rectangular plates with low height)
    # For ROI with height < 50 pixels, characters may appear wider and take up more height
    if h < 50:
        # Wider aspect ratio for small ROIs (characters look wider when compressed)
        aspect_ratio_range = (0.1, 3.0)  # Allow wider characters
        # Allow taller characters in small ROIs
        max_height_ratio = 0.98
        # Lower min_area_ratio for small ROIs (characters are also small in pixel count)
        min_area_ratio = 0.002
    
    # Binarize using Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean morphology
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, 
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
    # Invert if needed (characters should be white on black)
    inverted = cv2.bitwise_not(binary)
    
    # Try both binary and inverted, pick the one with more valid components
    candidates_binary = _find_char_components(
        binary, h, w, roi_area,
        min_area_ratio, max_area_ratio, aspect_ratio_range,
        min_height_ratio, max_height_ratio
    )
    candidates_inverted = _find_char_components(
        inverted, h, w, roi_area,
        min_area_ratio, max_area_ratio, aspect_ratio_range,
        min_height_ratio, max_height_ratio
    )
    
    # Choose the version with more valid components
    if len(candidates_inverted) > len(candidates_binary):
        candidates = candidates_inverted
        used_inverted = True
    else:
        candidates = candidates_binary
        used_inverted = False
    
    debug_info = {
        "roi_size": (w, h),
        "binary_candidates": len(candidates_binary),
        "inverted_candidates": len(candidates_inverted),
        "used_inverted": used_inverted,
        "total_candidates": len(candidates)
    }
    
    return candidates, debug_info


def _find_char_components(
    mask: np.ndarray,
    roi_h: int,
    roi_w: int,
    roi_area: int,
    min_area_ratio: float,
    max_area_ratio: float,
    aspect_ratio_range: Tuple[float, float],
    min_height_ratio: float,
    max_height_ratio: float
) -> List[Tuple[int, int, int, int]]:
    """Find character-like connected components in a binary mask."""
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    candidates = []
    min_area = roi_area * min_area_ratio
    max_area = roi_area * max_area_ratio
    min_height = roi_h * min_height_ratio
    max_height = roi_h * max_height_ratio
    
    for i in range(1, num_labels):  # Skip background (0)
        x, y, bw, bh, area = stats[i]
        
        # Area filter
        if area < min_area or area > max_area:
            continue
        
        # Height filter
        if bh < min_height or bh > max_height:
            continue
        
        # Aspect ratio filter
        aspect = bw / float(bh) if bh > 0 else 0
        if not (aspect_ratio_range[0] <= aspect <= aspect_ratio_range[1]):
            continue
        
        # Margin filter (characters shouldn't touch edges)
        margin = min(5, roi_w // 20, roi_h // 20)
        if x < margin and (x + bw) > (roi_w - margin):
            continue  # Spans entire width - likely border
        
        candidates.append((x, y, bw, bh))
    
    return candidates


def compute_character_score(
    roi_image: np.ndarray,
    expected_chars_range: Tuple[int, int] = (4, 10),
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict]:
    """Compute a validation score based on character-like components.
    
    License plates should contain a specific number of characters with
    consistent sizing and spacing. This function scores how likely
    the ROI contains a valid plate based on detected characters.
    
    Supports both single-line and two-line (square) Vietnamese plates.
    
    Args:
        roi_image: Grayscale or BGR image of the plate region
        expected_chars_range: Expected number of characters (min, max)
        weights: Optional weights for different scoring factors
    
    Returns:
        (score, details) where score is 0-1 and details has component info
    """
    if weights is None:
        # Increased char_count weight - more chars = more likely real plate
        weights = {
            "char_count": 0.55,  # Increased from 0.4
            "size_consistency": 0.20,  # Reduced from 0.3
            "spacing_regularity": 0.15,  # Reduced from 0.2
            "alignment": 0.10
        }
    
    # Detect character candidates
    chars, detect_info = detect_character_candidates(roi_image)
    
    if len(chars) == 0:
        return 0.0, {"chars": 0, "reason": "no_characters"}
    
    # Check if this might be a two-line plate (square plate)
    # by analyzing the vertical distribution of characters
    gray = ensure_grayscale(roi_image)
    roi_h, roi_w = gray.shape[:2]
    aspect_ratio = roi_w / roi_h if roi_h > 0 else 0
    
    # Two-line plate detection: AR close to 1 and characters in 2 distinct rows
    is_two_line = False
    line1_chars = []
    line2_chars = []
    
    if 0.5 < aspect_ratio < 2.0 and len(chars) >= 3:
        # Check if characters form 2 distinct horizontal lines
        y_centers = sorted([y + h/2 for (_, y, _, h) in chars])
        
        if len(y_centers) >= 3:
            # Find the largest gap in y-centers
            gaps = [(y_centers[i+1] - y_centers[i], i) for i in range(len(y_centers)-1)]
            if gaps:
                max_gap, gap_idx = max(gaps, key=lambda x: x[0])
                
                # If max gap is significant (> 20% of ROI height), it's likely 2 lines
                if max_gap > roi_h * 0.15:
                    is_two_line = True
                    threshold_y = (y_centers[gap_idx] + y_centers[gap_idx + 1]) / 2
                    
                    for (x, y, w, h) in chars:
                        center_y = y + h/2
                        if center_y < threshold_y:
                            line1_chars.append((x, y, w, h))
                        else:
                            line2_chars.append((x, y, w, h))
    
    # Score 1: Character count (prefer 6-8 chars for VN plates)
    # VN plates typically have 7-9 chars (e.g., 51F-86947 or 76A\n222.22)
    min_chars, max_chars = expected_chars_range
    char_count_score = 0.0
    ideal_chars = 7  # Ideal for VN plates
    
    if is_two_line:
        # For two-line plates, expect 3-4 chars on line 1, 4-6 chars on line 2
        total_chars = len(chars)
        if 6 <= total_chars <= 10:
            # Good count for 2-line plate
            if len(line1_chars) >= 2 and len(line2_chars) >= 3:
                char_count_score = 0.95  # High score for valid 2-line structure
            else:
                char_count_score = 0.8
        elif total_chars >= min_chars:
            char_count_score = 0.7
        else:
            char_count_score = total_chars / min_chars * 0.5
    else:
        if min_chars <= len(chars) <= max_chars:
            # Give bonus for being close to ideal
            distance_from_ideal = abs(len(chars) - ideal_chars)
            char_count_score = max(0.7, 1.0 - distance_from_ideal * 0.1)
        elif len(chars) < min_chars:
            char_count_score = len(chars) / min_chars * 0.5  # Penalize more
        else:
            char_count_score = max(0, 1.0 - (len(chars) - max_chars) / max_chars * 0.5)
    
    # Score 2: Size consistency (characters should be similar size)
    heights = [h for (_, _, _, h) in chars]
    widths = [w for (_, _, w, _) in chars]
    
    if is_two_line and len(line1_chars) > 0 and len(line2_chars) > 0:
        # For 2-line plates, check consistency within each line
        heights1 = [h for (_, _, _, h) in line1_chars]
        heights2 = [h for (_, _, _, h) in line2_chars]
        
        std1 = np.std(heights1) / np.mean(heights1) if len(heights1) > 1 and np.mean(heights1) > 0 else 0
        std2 = np.std(heights2) / np.mean(heights2) if len(heights2) > 1 and np.mean(heights2) > 0 else 0
        size_score = max(0, 1.0 - (std1 + std2) / 2)
    elif len(heights) > 1:
        height_std = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 1
        width_std = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 1
        size_score = max(0, 1.0 - (height_std + width_std) / 2)
    else:
        size_score = 0.5  # Single character - neutral score
    
    # Score 3: Spacing regularity (gaps between characters should be regular)
    if is_two_line:
        # For 2-line plates, check spacing within each line
        spacing_scores = []
        for line_chars in [line1_chars, line2_chars]:
            if len(line_chars) > 2:
                chars_sorted = sorted(line_chars, key=lambda c: c[0])
                gaps = []
                for i in range(len(chars_sorted) - 1):
                    x1, _, w1, _ = chars_sorted[i]
                    x2, _, _, _ = chars_sorted[i + 1]
                    gap = x2 - (x1 + w1)
                    if gap > 0:
                        gaps.append(gap)
                if gaps:
                    gap_std = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 1
                    spacing_scores.append(max(0, 1.0 - gap_std))
        spacing_score = np.mean(spacing_scores) if spacing_scores else 0.5
    else:
        chars_sorted = sorted(chars, key=lambda c: c[0])
        if len(chars_sorted) > 2:
            gaps = []
            for i in range(len(chars_sorted) - 1):
                x1, _, w1, _ = chars_sorted[i]
                x2, _, _, _ = chars_sorted[i + 1]
                gap = x2 - (x1 + w1)
                if gap > 0:
                    gaps.append(gap)
        
            if gaps:
                gap_std = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 1
                spacing_score = max(0, 1.0 - gap_std)
            else:
                spacing_score = 0.3
        else:
            spacing_score = 0.5
    
    # Score 4: Vertical alignment (characters should be roughly aligned)
    if is_two_line:
        # For 2-line plates, check alignment within each line separately
        alignment_scores = []
        for line_chars in [line1_chars, line2_chars]:
            if len(line_chars) > 1:
                line_heights = [h for (_, _, _, h) in line_chars]
                y_centers = [y + h/2 for (_, y, _, h) in line_chars]
                y_std = np.std(y_centers)
                avg_height = np.mean(line_heights)
                if avg_height > 0:
                    alignment_scores.append(max(0, 1.0 - y_std / avg_height))
        alignment_score = np.mean(alignment_scores) if alignment_scores else 0.5
    else:
        if len(chars) > 1:
            y_centers = [y + h/2 for (_, y, _, h) in chars]
            y_std = np.std(y_centers)
            avg_height = np.mean(heights)
            alignment_score = max(0, 1.0 - y_std / avg_height) if avg_height > 0 else 0
        else:
            alignment_score = 0.5
    
    # Compute weighted total
    total_score = (
        weights["char_count"] * char_count_score +
        weights["size_consistency"] * size_score +
        weights["spacing_regularity"] * spacing_score +
        weights["alignment"] * alignment_score
    )
    
    # Bonus for valid 2-line structure
    if is_two_line and len(line1_chars) >= 2 and len(line2_chars) >= 3:
        total_score = min(1.0, total_score * 1.1)  # 10% bonus
    
    details = {
        "char_count": len(chars),
        "char_boxes": chars,
        "is_two_line": is_two_line,
        "line1_count": len(line1_chars) if is_two_line else 0,
        "line2_count": len(line2_chars) if is_two_line else 0,
        "scores": {
            "char_count": char_count_score,
            "size_consistency": size_score,
            "spacing_regularity": spacing_score,
            "alignment": alignment_score
        },
        "total_score": total_score,
        "detect_info": detect_info
    }
    
    return total_score, details


def validate_plate_roi(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    min_score: float = 0.4,
    expected_chars: Tuple[int, int] = (5, 10)
) -> Tuple[bool, float, Dict]:
    """Validate a detected plate ROI using character analysis.
    
    Args:
        image: Full image (grayscale or BGR)
        roi: Plate bounding box (x, y, w, h)
        min_score: Minimum score to consider valid
        expected_chars: Expected character count range
    
    Returns:
        (is_valid, score, details)
    """
    x, y, w, h = roi
    gray = ensure_grayscale(image)
    
    # Extract ROI with bounds checking
    img_h, img_w = gray.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    
    plate_roi = gray[y1:y2, x1:x2]
    
    if plate_roi.size == 0:
        return False, 0.0, {"error": "Empty ROI"}
    
    score, details = compute_character_score(plate_roi, expected_chars)
    is_valid = score >= min_score
    
    details["is_valid"] = is_valid
    details["min_score"] = min_score
    
    return is_valid, score, details


def filter_detections_by_characters(
    image: np.ndarray,
    detections: List[Dict],
    min_char_score: float = 0.35,
    expected_chars: Tuple[int, int] = (5, 10),
    keep_invalid: bool = False
) -> Tuple[List[Dict], Dict]:
    """Filter plate detections using character-based validation.
    
    Args:
        image: Full image
        detections: List of detection dicts with 'box' key
        min_char_score: Minimum character score to keep
        expected_chars: Expected character count range
        keep_invalid: If True, keep invalid detections but mark them
    
    Returns:
        (filtered_detections, stats)
    """
    gray = ensure_grayscale(image)
    
    valid_detections = []
    stats = {
        "total_input": len(detections),
        "valid": 0,
        "invalid": 0,
        "scores": []
    }
    
    for det in detections:
        roi = det.get("box")
        if roi is None:
            continue
        
        is_valid, score, details = validate_plate_roi(
            gray, roi, 
            min_score=min_char_score,
            expected_chars=expected_chars
        )
        
        # Add validation info to detection
        det_copy = det.copy()
        det_copy["char_score"] = score
        det_copy["char_count"] = details.get("char_count", 0)
        det_copy["char_valid"] = is_valid
        det_copy["char_details"] = details
        
        stats["scores"].append(score)
        
        if is_valid:
            stats["valid"] += 1
            valid_detections.append(det_copy)
        else:
            stats["invalid"] += 1
            if keep_invalid:
                valid_detections.append(det_copy)
    
    return valid_detections, stats


# ============================================================================
# PHASE 7: COLOR-BASED AND MSER PLATE DETECTION (for difficult images)
# ============================================================================

def detect_white_plate_regions(
    image: np.ndarray,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.25,
    aspect_ratio_range: Tuple[float, float] = (0.8, 5.0),
    debug: bool = False
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Detect white/light colored plate regions using color thresholding.
    
    Vietnamese plates are typically white (car) or yellow (taxi/commercial).
    This helps detect plates in complex backgrounds like chrome grills.
    
    Args:
        image: BGR or grayscale image
        min_area_ratio: Minimum region area as ratio of image area
        max_area_ratio: Maximum region area as ratio of image area
        aspect_ratio_range: Valid aspect ratio range for plates
        debug: Return debug information
    
    Returns:
        (plate_boxes, debug_info)
    """
    # Ensure BGR image for color analysis
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()
    
    h, w = bgr.shape[:2]
    img_area = h * w
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) == 3 else bgr
    
    # White plate detection: high value, low saturation
    # HSV ranges for white: H=any, S=0-50, V=180-255
    white_lower = np.array([0, 0, 180], dtype=np.uint8)
    white_upper = np.array([180, 60, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Yellow plate detection (for taxi/commercial)
    # HSV ranges for yellow: H=15-35, S=80-255, V=150-255
    yellow_lower = np.array([15, 80, 150], dtype=np.uint8)
    yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Combine masks
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Clean up mask with morphological operations
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Find contours in color mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plates = []
    rejected = {"area": 0, "aspect": 0, "solidity": 0}
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            rejected["area"] += 1
            continue
        
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / float(bh) if bh > 0 else 0
        
        if not (aspect_ratio_range[0] <= aspect <= aspect_ratio_range[1]):
            rejected["aspect"] += 1
            continue
        
        # Check solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if solidity < 0.4:
            rejected["solidity"] += 1
            continue
        
        plates.append((x, y, bw, bh))
    
    debug_info = {
        "method": "color_white_yellow",
        "contours_found": len(contours),
        "rejected": rejected,
        "white_mask": white_mask if debug else None,
        "yellow_mask": yellow_mask if debug else None,
        "combined_mask": color_mask if debug else None
    }
    
    return plates, debug_info


def detect_with_mser(
    gray_image: np.ndarray,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.25,
    aspect_ratio_range: Tuple[float, float] = (0.8, 5.0),
    debug: bool = False
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Detect plate regions using MSER (Maximally Stable Extremal Regions).
    
    MSER is good at finding text-like regions because text characters
    typically form stable regions with consistent intensity.
    
    Args:
        gray_image: Grayscale image
        min_area_ratio: Minimum region area as ratio of image area
        max_area_ratio: Maximum region area as ratio of image area
        aspect_ratio_range: Valid aspect ratio range for plates
        debug: Return debug information
    
    Returns:
        (plate_boxes, debug_info)
    """
    gray = ensure_grayscale(gray_image)
    h, w = gray.shape[:2]
    img_area = h * w
    
    # Create MSER detector
    mser = cv2.MSER_create(
        delta=5,
        min_area=60,
        max_area=int(img_area * 0.1),
        max_variation=0.25,
        min_diversity=0.2,
        max_evolution=200,
        area_threshold=1.01,
        min_margin=0.003,
        edge_blur_size=5
    )
    
    # Detect MSER regions
    regions, _ = mser.detectRegions(gray)
    
    if len(regions) == 0:
        return [], {"method": "mser", "regions_found": 0}
    
    # Convert regions to bounding boxes
    region_boxes = []
    for region in regions:
        x, y, bw, bh = cv2.boundingRect(region)
        region_boxes.append((x, y, bw, bh))
    
    # Group nearby boxes (text regions tend to cluster for plate)
    # Create mask of MSER regions
    mser_mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        cv2.fillPoly(mser_mask, [region], 255)
    
    # Dilate to connect nearby text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    dilated = cv2.dilate(mser_mask, kernel, iterations=2)
    
    # Close gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    
    # Find contours of grouped regions
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plates = []
    rejected = {"area": 0, "aspect": 0, "density": 0}
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio
    
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        
        if not (min_area <= area <= max_area):
            rejected["area"] += 1
            continue
        
        aspect = bw / float(bh) if bh > 0 else 0
        if not (aspect_ratio_range[0] <= aspect <= aspect_ratio_range[1]):
            rejected["aspect"] += 1
            continue
        
        # Check MSER density in this region (should have text-like content)
        roi_mask = mser_mask[y:y+bh, x:x+bw]
        density = np.sum(roi_mask > 0) / float(area) if area > 0 else 0
        
        if density < 0.15:  # At least 15% MSER content
            rejected["density"] += 1
            continue
        
        plates.append((x, y, bw, bh))
    
    debug_info = {
        "method": "mser",
        "regions_found": len(regions),
        "contours_found": len(contours),
        "rejected": rejected,
        "mser_mask": mser_mask if debug else None,
        "dilated": dilated if debug else None
    }
    
    return plates, debug_info


def detect_with_all_methods(
    image: np.ndarray,
    presets: Optional[List[str]] = None,
    use_color: bool = True,
    use_mser: bool = True,
    debug: bool = False
) -> Tuple[List[Dict], Dict]:
    """Comprehensive detection using multiple methods with fusion.
    
    Combines:
    1. Contour-based multi-preset detection
    2. Edge density sliding window (backup)
    3. Color-based detection (white/yellow plates)
    4. MSER text region detection
    
    Results are fused using NMS and character validation scoring.
    
    Args:
        image: BGR or grayscale image
        presets: Presets for contour detection
        use_color: Whether to use color-based detection
        use_mser: Whether to use MSER detection
        debug: Return debug information
    
    Returns:
        (detections, debug_info)
    """
    # Prepare images
    if len(image.shape) == 3:
        bgr = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    all_candidates = []
    debug_info = {"methods_used": []}
    
    # Method 1: Contour-based with edge backup
    contour_results, contour_debug = detect_with_edge_backup(
        gray, presets=presets, debug=debug
    )
    for det in contour_results:
        det["source"] = "contour"
        all_candidates.append(det)
    debug_info["contour"] = contour_debug
    debug_info["methods_used"].append("contour")
    
    # Method 2: Color-based detection
    if use_color:
        color_plates, color_debug = detect_white_plate_regions(bgr, debug=debug)
        for plate in color_plates:
            x, y, w, h = plate
            all_candidates.append({
                "box": plate,
                "method": "color",
                "source": "color",
                "score": 0.8,  # Lower initial score
                "aspect_ratio": w / float(h) if h > 0 else 0
            })
        debug_info["color"] = color_debug
        debug_info["methods_used"].append("color")
    
    # Method 3: MSER text region detection
    if use_mser:
        mser_plates, mser_debug = detect_with_mser(gray, debug=debug)
        for plate in mser_plates:
            x, y, w, h = plate
            all_candidates.append({
                "box": plate,
                "method": "mser",
                "source": "mser",
                "score": 0.7,  # Lower initial score
                "aspect_ratio": w / float(h) if h > 0 else 0
            })
        debug_info["mser"] = mser_debug
        debug_info["methods_used"].append("mser")
    
    # Remove duplicates using NMS
    if all_candidates:
        boxes = [c["box"] for c in all_candidates]
        scores = [c.get("score", 0.5) for c in all_candidates]
        keep_boxes = non_maximum_suppression(boxes, scores, iou_threshold=0.3)
        keep_set = set(keep_boxes)
        all_candidates = [c for c in all_candidates if c["box"] in keep_set]
    
    debug_info["total_candidates"] = len(all_candidates)
    
    return all_candidates, debug_info


def detect_with_character_validation(
    gray_image: np.ndarray,
    bgr_image: np.ndarray = None,
    presets: Optional[List[str]] = None,
    use_edge_backup: bool = True,
    use_color: bool = False,
    use_mser: bool = False,
    min_char_score: float = 0.35,
    expected_chars: Tuple[int, int] = (5, 10),
    debug: bool = False
) -> Tuple[List[Dict], Dict]:
    """Full detection pipeline with character-based validation.
    
    This combines:
    1. Multi-preset contour detection
    2. Optional edge density backup
    3. Optional color-based detection
    4. Optional MSER text region detection
    5. Character-based ROI validation
    
    Args:
        gray_image: Grayscale image
        bgr_image: Original BGR image (needed for color detection)
        presets: Detection presets to use
        use_edge_backup: Whether to use edge density backup
        use_color: Whether to use color-based detection (for difficult images)
        use_mser: Whether to use MSER detection (for difficult images)
        min_char_score: Minimum character validation score
        expected_chars: Expected character count range
        debug: Return debug info
    
    Returns:
        (valid_detections, debug_info)
    """
    gray = ensure_grayscale(gray_image)
    
    # Step 1: Detect candidates using selected methods
    if use_color or use_mser:
        # Use comprehensive multi-method detection
        # Pass BGR image if available, otherwise pass gray (will be converted)
        input_image = bgr_image if bgr_image is not None else gray
        candidates, detect_info = detect_with_all_methods(
            input_image, presets=presets, 
            use_color=use_color, use_mser=use_mser,
            debug=debug
        )
    elif use_edge_backup:
        candidates, detect_info = detect_with_edge_backup(gray, presets=presets, debug=debug)
    else:
        plates, detect_info = detect_multi_preset(gray, presets=presets, debug=debug)
        candidates = [{"box": p, "method": "contour", "score": 1.0} for p in plates]
    
    # Step 2: Validate with character analysis
    valid_detections, char_stats = filter_detections_by_characters(
        gray,
        candidates,
        min_char_score=min_char_score,
        expected_chars=expected_chars,
        keep_invalid=debug
    )
    
    debug_info = {
        "detection_info": detect_info if debug else None,
        "char_validation": char_stats,
        "candidates_count": len(candidates),
        "valid_count": char_stats["valid"]
    }
    
    # Return only valid detections (or all if debug)
    if debug:
        return valid_detections, debug_info
    else:
        return [d for d in valid_detections if d.get("char_valid", False)], debug_info

