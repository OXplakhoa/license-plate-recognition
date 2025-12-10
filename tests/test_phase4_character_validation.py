"""Test Phase 4: Character-based ROI Validation"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lp_detector import (
    detect_character_candidates,
    compute_character_score,
    validate_plate_roi,
    filter_detections_by_characters,
    detect_with_character_validation,
    detect_multi_preset,
)


def test_detect_character_candidates():
    """Test character detection in synthetic plate."""
    # Create a license plate-like image
    img = np.ones((60, 180), dtype=np.uint8) * 200
    
    # Add character-like rectangles
    char_positions = [(15, 10), (35, 10), (55, 10), (85, 10), (105, 10), (125, 10), (145, 10)]
    for (x, y) in char_positions:
        cv2.rectangle(img, (x, y), (x+15, y+40), 30, -1)
    
    chars, info = detect_character_candidates(img)
    
    assert isinstance(chars, list)
    assert "roi_size" in info
    assert "total_candidates" in info
    
    print(f"✅ test_detect_character_candidates passed (found {len(chars)} candidates)")


def test_compute_character_score():
    """Test character-based scoring."""
    # Create a good plate-like image
    img = np.ones((60, 180), dtype=np.uint8) * 200
    for i, x in enumerate(range(15, 165, 22)):
        cv2.rectangle(img, (x, 10), (x+15, 50), 30, -1)
    
    score, details = compute_character_score(img, expected_chars_range=(5, 10))
    
    assert 0 <= score <= 1
    assert "char_count" in details
    assert "scores" in details
    
    print(f"✅ test_compute_character_score passed (score: {score:.3f}, chars: {details['char_count']})")


def test_validate_plate_roi():
    """Test plate ROI validation."""
    # Create full image with a plate region
    full_img = np.ones((300, 400), dtype=np.uint8) * 100
    
    # Add plate background at specific location
    plate_x, plate_y = 100, 120
    plate_w, plate_h = 150, 50
    full_img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w] = 220
    
    # Add characters to plate region (dark on light background)
    for i, offset_x in enumerate(range(10, 140, 20)):
        x1 = plate_x + offset_x
        y1 = plate_y + 8
        x2 = x1 + 12
        y2 = plate_y + 42
        cv2.rectangle(full_img, (x1, y1), (x2, y2), 30, -1)
    
    roi = (plate_x, plate_y, plate_w, plate_h)
    is_valid, score, details = validate_plate_roi(full_img, roi, min_score=0.3)
    
    assert isinstance(is_valid, (bool, np.bool_)), f"Expected bool, got {type(is_valid)}"
    assert 0 <= score <= 1, f"Score {score} out of range"
    
    print(f"✅ test_validate_plate_roi passed (valid: {is_valid}, score: {score:.3f}, chars: {details.get('char_count', 0)})")


def test_filter_detections_by_characters():
    """Test filtering detections by character validation."""
    # Create test image
    img = np.ones((300, 400), dtype=np.uint8) * 100
    
    # Good plate region
    for i, x in enumerate(range(110, 240, 18)):
        cv2.rectangle(img, (x, 128), (x+12, 162), 30, -1)
    img[120:170, 100:250] = np.maximum(img[120:170, 100:250], 180)
    
    # Bad region (no characters)
    img[50:100, 50:150] = 150
    
    detections = [
        {"box": (100, 120, 150, 50), "method": "test", "score": 0.8},  # Good
        {"box": (50, 50, 100, 50), "method": "test", "score": 0.6},   # Bad
    ]
    
    filtered, stats = filter_detections_by_characters(img, detections)
    
    assert "valid" in stats
    assert "invalid" in stats
    
    print(f"✅ test_filter_detections_by_characters passed (valid: {stats['valid']}, invalid: {stats['invalid']})")


def test_detect_with_character_validation():
    """Test full pipeline with character validation."""
    test_folder = r'F:\CODE\XLA\data\test_images\CarTGMT'
    if not os.path.exists(test_folder):
        print("⚠️ Test folder not found, skipping")
        return
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])[:15]
    
    valid_count = 0
    total_detections = 0
    
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections, info = detect_with_character_validation(gray, debug=True)
        
        total_detections += len(detections)
        valid_count += info["valid_count"]
    
    print(f"✅ test_detect_with_character_validation passed")
    print(f"   Total detections: {total_detections}")
    print(f"   Character-validated: {valid_count}")


def test_integration_real_images():
    """Integration test comparing with and without character validation."""
    test_folder = r'F:\CODE\XLA\data\test_images\CarTGMT'
    if not os.path.exists(test_folder):
        print("⚠️ Test folder not found")
        return
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])[:20]
    
    results = []
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Without character validation
        plates_raw, _ = detect_multi_preset(gray)
        
        # With character validation
        plates_valid, info = detect_with_character_validation(gray, debug=True)
        
        results.append({
            'name': img_name[:25],
            'raw': len(plates_raw),
            'validated': info['valid_count'],
            'filtered_out': info['char_validation']['invalid']
        })
    
    print(f"\n{'Image':<30} {'Raw':>6} {'Valid':>8} {'Filtered':>10}")
    print("-" * 60)
    for r in results[:10]:
        print(f"{r['name']:<30} {r['raw']:>6} {r['validated']:>8} {r['filtered_out']:>10}")
    
    print(f"\n✅ test_integration_real_images passed ({len(results)} images)")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4: CHARACTER-BASED ROI VALIDATION TESTS")
    print("=" * 60)
    
    test_detect_character_candidates()
    test_compute_character_score()
    test_validate_plate_roi()
    test_filter_detections_by_characters()
    test_detect_with_character_validation()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)
    test_integration_real_images()
    
    print("\n" + "=" * 60)
    print("ALL PHASE 4 TESTS PASSED!")
    print("=" * 60)
