"""
Test multi-preset detection functionality.
"""
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.lp_detector import (
    detect_multi_preset,
    detect_license_plates,
    compute_iou,
    non_maximum_suppression,
    merge_overlapping_boxes,
    classify_detected_plates,
    get_plate_type_from_aspect,
    PRESETS,
    DEFAULT_MULTI_PRESETS,
)
from src.utils import ensure_grayscale


def load_and_preprocess(image_path):
    """Load and preprocess an image for testing."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return ensure_grayscale(img)


def test_presets_exist():
    """Test that all required presets are defined."""
    required = ["car_square", "car_rect", "bike"]
    for preset in required:
        assert preset in PRESETS, f"Missing preset: {preset}"
    print("✅ All required presets exist")


def test_compute_iou():
    """Test IoU computation."""
    # No overlap
    iou = compute_iou((0, 0, 10, 10), (100, 100, 10, 10))
    assert iou == 0.0, f"Expected 0, got {iou}"
    
    # Full overlap
    iou = compute_iou((0, 0, 10, 10), (0, 0, 10, 10))
    assert iou == 1.0, f"Expected 1, got {iou}"
    
    # Partial overlap
    iou = compute_iou((0, 0, 10, 10), (5, 5, 10, 10))
    assert 0 < iou < 1, f"Expected partial overlap, got {iou}"
    
    print("✅ IoU computation works correctly")


def test_nms():
    """Test non-maximum suppression."""
    boxes = [(0, 0, 100, 100), (10, 10, 100, 100), (200, 200, 50, 50)]
    result = non_maximum_suppression(boxes, iou_threshold=0.3)
    
    # Should keep 2 boxes (merge overlapping, keep separate)
    assert len(result) == 2, f"Expected 2 boxes, got {len(result)}"
    print("✅ NMS works correctly")


def test_merge_boxes():
    """Test box merging."""
    boxes = [(0, 0, 100, 100), (50, 50, 100, 100)]
    result = merge_overlapping_boxes(boxes, iou_threshold=0.1)
    
    # Should merge into one larger box
    assert len(result) == 1, f"Expected 1 merged box, got {len(result)}"
    
    # Merged box should encompass both
    x, y, w, h = result[0]
    assert x == 0 and y == 0, "Merged box should start at origin"
    assert x + w >= 150 and y + h >= 150, "Merged box should cover both inputs"
    print("✅ Box merging works correctly")


def test_plate_classification():
    """Test plate type classification by aspect ratio and size."""
    # car2 (2-line): AR < 1.4
    assert get_plate_type_from_aspect(1.2) == "car2"
    
    # car1 (1-line): AR > 2.5
    assert get_plate_type_from_aspect(4.0) == "car1"
    
    # Intermediate AR (1.4-2.5): depends on size
    # Without size info (default), uses AR boundary
    assert get_plate_type_from_aspect(2.0) == "bike"  # No size info → bike
    
    # With size info: large plate → car2
    assert get_plate_type_from_aspect(1.8, width=150, height=80) == "car2"
    
    # With size info: small plate → bike
    assert get_plate_type_from_aspect(1.8, width=100, height=60) == "bike"
    
    print("✅ Plate classification works correctly")


def test_detect_multi_preset_synthetic():
    """Test multi-preset detection with synthetic image."""
    # Create a simple synthetic image with a rectangle
    img = np.zeros((400, 600), dtype=np.uint8)
    
    # Draw a rectangle that looks like a plate
    cv2.rectangle(img, (200, 150), (400, 250), 255, -1)
    
    plates, info = detect_multi_preset(img, debug=True)
    
    assert "preset_results" in info
    assert "total_candidates" in info
    assert isinstance(plates, list)
    
    print(f"✅ Multi-preset detection runs (found {len(plates)} plates)")


def test_detect_with_real_image():
    """Test detection with a real image if available."""
    test_path = r"F:\CODE\XLA\data\test_images\CarTGMT"
    
    if not os.path.exists(test_path):
        print("⚠️ Skipping real image test - path not found")
        return
    
    # Get first image
    images = [f for f in os.listdir(test_path) if f.endswith('.jpg')]
    if not images:
        print("⚠️ Skipping real image test - no images found")
        return
    
    img_path = os.path.join(test_path, images[0])
    gray = load_and_preprocess(img_path)
    
    if gray is None:
        print("⚠️ Skipping real image test - preprocessing failed")
        return
    
    # Compare single vs multi
    plates_single, _ = detect_license_plates(gray, mode="car_square")
    plates_multi, info = detect_multi_preset(gray, debug=True)
    
    print(f"✅ Real image test: single={len(plates_single)}, multi={len(plates_multi)}")
    print(f"   Candidates from presets: {info['total_candidates']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multi-Preset Detection")
    print("=" * 60)
    
    test_presets_exist()
    test_compute_iou()
    test_nms()
    test_merge_boxes()
    test_plate_classification()
    test_detect_multi_preset_synthetic()
    test_detect_with_real_image()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
