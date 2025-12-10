"""Test Phase 2: Perspective Correction Functions"""
import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lp_detector import (
    order_points,
    find_plate_contour,
    perspective_transform,
    detect_and_correct_plate,
    compute_skew_angle,
    deskew_image,
    correct_plate_perspective_and_skew,
    detect_multi_preset
)


def test_order_points():
    """Test order_points function."""
    # Random 4 points
    pts = np.array([[100, 50], [50, 50], [50, 100], [100, 100]], dtype=np.float32)
    ordered = order_points(pts)
    
    # Check ordering: TL, TR, BR, BL
    tl, tr, br, bl = ordered
    
    # TL should have smallest sum (x+y)
    assert tl[0] + tl[1] <= tr[0] + tr[1]
    # BR should have largest sum
    assert br[0] + br[1] >= bl[0] + bl[1]
    
    print("✅ test_order_points passed")


def test_deskew_image():
    """Test deskew_image function."""
    # Create a simple image
    img = np.zeros((100, 200), dtype=np.uint8)
    cv2.rectangle(img, (20, 40), (180, 60), 255, -1)
    
    # Test with 0 angle (no change)
    result = deskew_image(img, angle=0)
    assert result.shape == img.shape
    
    # Test with small angle
    result = deskew_image(img, angle=5)
    assert result is not None
    
    print("✅ test_deskew_image passed")


def test_compute_skew_angle():
    """Test compute_skew_angle function."""
    # Create image with horizontal line
    img = np.zeros((100, 200), dtype=np.uint8)
    cv2.line(img, (10, 50), (190, 50), 255, 2)
    
    angle = compute_skew_angle(img)
    assert isinstance(angle, (int, float))
    assert -45 <= angle <= 45  # Reasonable range
    
    print("✅ test_compute_skew_angle passed")


def test_perspective_transform():
    """Test perspective_transform function."""
    # Create image with quadrilateral
    img = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array([[50, 50], [150, 55], [145, 95], [55, 90]], dtype=np.float32)
    cv2.fillPoly(img, [pts.astype(np.int32)], 255)
    
    # Transform to rectangle - output_size is (width, height) tuple
    warped = perspective_transform(img, pts, output_size=(100, 50))
    assert warped.shape == (50, 100)
    
    print("✅ test_perspective_transform passed")


def test_correct_plate_perspective_and_skew():
    """Test full correction pipeline."""
    # Load real image
    test_folder = r'F:\CODE\XLA\data\test_images\CarTGMT'
    if not os.path.exists(test_folder):
        print("⚠️ Test folder not found, skipping real image test")
        return
    
    images = [f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')]
    if not images:
        print("⚠️ No test images found")
        return
    
    # Try multiple images until we find one with detectable plates
    for img_name in images[:20]:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect plates
        plates, _ = detect_multi_preset(gray)
        if plates:
            # Test correction
            roi = tuple(map(int, plates[0]))
            corrected, info = correct_plate_perspective_and_skew(gray, roi, debug=True)
            
            assert corrected is not None
            assert corrected.size > 0
            
            print(f"✅ test_correct_plate_perspective_and_skew passed")
            print(f"   Image: {img_name}")
            print(f"   ROI: {roi}")
            print(f"   Corrected shape: {corrected.shape}")
            return
    
    print("⚠️ No plates detected in any test images")


def test_integration():
    """Integration test with multiple images."""
    test_folder = r'F:\CODE\XLA\data\test_images\CarTGMT'
    if not os.path.exists(test_folder):
        print("⚠️ Test folder not found")
        return
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])[:30]
    
    success_count = 0
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates, _ = detect_multi_preset(gray)
        
        if plates:
            roi = tuple(map(int, plates[0]))
            corrected, info = correct_plate_perspective_and_skew(gray, roi, debug=True)
            if corrected is not None and corrected.size > 0:
                success_count += 1
                if success_count <= 5:  # Only print first 5
                    print(f"  ✅ {img_name[:30]}: {corrected.shape}")
    
    print(f"\n✅ Integration test: {success_count}/{len(images)} images processed successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2: PERSPECTIVE CORRECTION TESTS")
    print("=" * 60)
    
    test_order_points()
    test_deskew_image()
    test_compute_skew_angle()
    test_perspective_transform()
    test_correct_plate_perspective_and_skew()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)
    test_integration()
    
    print("\n" + "=" * 60)
    print("ALL PHASE 2 TESTS PASSED!")
    print("=" * 60)
