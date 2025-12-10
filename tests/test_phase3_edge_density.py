"""Test Phase 3: Edge Density Backup Detector"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lp_detector import (
    compute_edge_density,
    compute_vertical_edge_score,
    compute_contrast_score,
    compute_plate_score,
    sliding_window_detect,
    detect_with_edge_backup,
)


def test_compute_edge_density():
    """Test edge density computation."""
    # High edge density image (many edges)
    img_high = np.zeros((100, 100), dtype=np.uint8)
    for i in range(0, 100, 5):
        cv2.line(img_high, (i, 0), (i, 100), 255, 1)
    
    # Low edge density image (few edges)
    img_low = np.ones((100, 100), dtype=np.uint8) * 128
    
    density_high = compute_edge_density(img_high)
    density_low = compute_edge_density(img_low)
    
    assert density_high > density_low, f"Expected high > low, got {density_high} vs {density_low}"
    assert 0 <= density_high <= 1
    assert 0 <= density_low <= 1
    
    print("✅ test_compute_edge_density passed")


def test_compute_vertical_edge_score():
    """Test vertical edge score computation."""
    # Image with vertical lines
    img_vert = np.zeros((100, 100), dtype=np.uint8)
    for i in range(0, 100, 10):
        cv2.line(img_vert, (i, 0), (i, 100), 255, 2)
    
    # Image with horizontal lines
    img_horiz = np.zeros((100, 100), dtype=np.uint8)
    for i in range(0, 100, 10):
        cv2.line(img_horiz, (0, i), (100, i), 255, 2)
    
    score_vert = compute_vertical_edge_score(img_vert)
    score_horiz = compute_vertical_edge_score(img_horiz)
    
    assert score_vert > score_horiz, f"Expected vert > horiz, got {score_vert} vs {score_horiz}"
    
    print("✅ test_compute_vertical_edge_score passed")


def test_compute_contrast_score():
    """Test contrast score computation."""
    # High contrast (black and white)
    img_high = np.zeros((100, 100), dtype=np.uint8)
    img_high[:50, :] = 255
    
    # Low contrast (uniform gray)
    img_low = np.ones((100, 100), dtype=np.uint8) * 128
    
    score_high = compute_contrast_score(img_high)
    score_low = compute_contrast_score(img_low)
    
    assert score_high > score_low, f"Expected high > low, got {score_high} vs {score_low}"
    
    print("✅ test_compute_contrast_score passed")


def test_compute_plate_score():
    """Test combined plate score."""
    # Create a license plate-like image
    img = np.ones((50, 150), dtype=np.uint8) * 200
    # Add some vertical lines (characters)
    for i in range(20, 140, 15):
        cv2.rectangle(img, (i, 10), (i+8, 40), 30, -1)
    
    total, scores = compute_plate_score(img)
    
    assert 0 <= total <= 1
    assert "edge_density" in scores
    assert "vertical_edges" in scores
    assert "contrast" in scores
    
    print(f"✅ test_compute_plate_score passed (score: {total:.3f})")


def test_sliding_window_detect():
    """Test sliding window detection."""
    # Create test image
    img = np.ones((300, 400), dtype=np.uint8) * 128
    
    # Add a license plate-like region
    plate_region = np.ones((50, 120), dtype=np.uint8) * 200
    for i in range(10, 110, 12):
        cv2.rectangle(plate_region, (i, 8), (i+8, 42), 30, -1)
    
    img[100:150, 150:270] = plate_region
    
    detections, debug_info = sliding_window_detect(
        img, 
        score_threshold=0.3,  # Lower threshold for synthetic image
        max_candidates=10,
        debug=True
    )
    
    assert isinstance(detections, list)
    assert "windows_checked" in debug_info
    
    print(f"✅ test_sliding_window_detect passed (found {len(detections)} candidates)")


def test_detect_with_edge_backup():
    """Test combined detection with backup."""
    test_folder = r'F:\CODE\XLA\data\test_images\CarTGMT'
    if not os.path.exists(test_folder):
        print("⚠️ Test folder not found, skipping")
        return
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])[:10]
    
    backup_used_count = 0
    total_detections = 0
    
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections, debug_info = detect_with_edge_backup(gray, debug=True)
        
        total_detections += len(detections)
        if debug_info["backup_used"]:
            backup_used_count += 1
    
    print(f"✅ test_detect_with_edge_backup passed")
    print(f"   Total detections: {total_detections}")
    print(f"   Backup used: {backup_used_count}/{len(images)} images")


def test_integration_real_images():
    """Integration test with real images."""
    test_folder = r'F:\CODE\XLA\data\test_images\CarTGMT'
    if not os.path.exists(test_folder):
        print("⚠️ Test folder not found")
        return
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])[:20]
    
    success = 0
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Test edge density on cropped regions
        h, w = gray.shape
        regions = [
            (w//4, h//4, w//2, h//2),  # Center
            (0, h//2, w//3, h//3),      # Bottom left
            (w//2, h//2, w//3, h//3),   # Bottom right
        ]
        
        for (x, y, rw, rh) in regions:
            if x + rw <= w and y + rh <= h:
                roi = gray[y:y+rh, x:x+rw]
                score, _ = compute_plate_score(roi)
                assert 0 <= score <= 1
        
        success += 1
    
    print(f"✅ test_integration_real_images passed ({success} images)")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: EDGE DENSITY DETECTOR TESTS")
    print("=" * 60)
    
    test_compute_edge_density()
    test_compute_vertical_edge_score()
    test_compute_contrast_score()
    test_compute_plate_score()
    test_sliding_window_detect()
    test_detect_with_edge_backup()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)
    test_integration_real_images()
    
    print("\n" + "=" * 60)
    print("ALL PHASE 3 TESTS PASSED!")
    print("=" * 60)
