"""
Detailed Pipeline Debug Script
==============================

This script tests the pipeline step by step to identify exactly where
OCR confidence is being lost.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lp_detector import detect_with_character_validation
from src.character_segmenter import segment_characters_multi_method, binarize_plate
from src.ocr_engine import (
    ocr_single_char, 
    ocr_plate_line, 
    ocr_plate_multi_psm,
    configure_tesseract,
    resize_for_ocr,
    binarize_for_ocr,
    pad_image,
    validate_vn_plate_format,
    format_plate_display,
)
from src.utils import letter_digit_whitelist


TEST_FOLDER = r'F:\CODE\XLA\data\test_images\CarTGMT'


def test_single_image_detailed(image_path: str):
    """Test a single image through each pipeline step."""
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot load: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_name = os.path.basename(image_path)[:30]
    
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS: {img_name}")
    print(f"Image size: {img.shape}")
    print(f"{'='*70}")
    
    # Step 1: Detection
    print(f"\n[STEP 1: DETECTION]")
    detections, detect_info = detect_with_character_validation(gray, min_char_score=0.3, debug=True)
    print(f"  Detections found: {len(detections)}")
    
    if not detections:
        print("  No detections!")
        return
    
    # Analyze each detection
    for i, det in enumerate(detections[:3]):
        x, y, w, h = det['box']
        char_score = det.get('char_score', 0)
        method = det.get('method', 'unknown')
        
        roi = gray[y:y+h, x:x+w]
        
        print(f"\n  Detection {i+1}:")
        print(f"    Box: ({x}, {y}, {w}, {h})")
        print(f"    Aspect ratio: {w/h:.2f}")
        print(f"    Char score: {char_score:.3f}")
        print(f"    Method: {method}")
        
        # Step 2: Segmentation
        print(f"\n  [STEP 2: SEGMENTATION]")
        seg_result = segment_characters_multi_method(roi, plate_type='car2')
        num_chars = len(seg_result.boxes)
        print(f"    Characters found: {num_chars}")
        
        if seg_result.boxes:
            heights = [b[3] for b in seg_result.boxes]
            widths = [b[2] for b in seg_result.boxes]
            print(f"    Char heights: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
            print(f"    Char widths: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
        
        # Step 3: OCR - Per character
        print(f"\n  [STEP 3: OCR - Per Character]")
        if seg_result.char_images:
            configure_tesseract()
            whitelist = letter_digit_whitelist(True)
            
            chars = []
            confs = []
            for j, char_img in enumerate(seg_result.char_images):
                ch, conf = ocr_single_char(char_img, whitelist=whitelist)
                chars.append(ch)
                confs.append(conf)
                print(f"    Char {j+1}: '{ch}' (conf: {conf:.1f}%)")
            
            text = ''.join(chars)
            mean_conf = np.mean([c for c in confs if c >= 0]) if confs else 0
            print(f"    Combined: '{text}' (mean conf: {mean_conf:.1f}%)")
        else:
            print(f"    No characters to OCR")
        
        # Step 4: OCR - Whole line
        print(f"\n  [STEP 4: OCR - Whole Line Multi-PSM]")
        result = ocr_plate_multi_psm(roi, vn_plate=True)
        print(f"    Result: '{result.text}' (mean conf: {result.mean_conf:.1f}%)")
        
        # Step 5: Validation
        print(f"\n  [STEP 5: VALIDATION]")
        if result.text:
            is_valid, reason = validate_vn_plate_format(result.text)
            display = format_plate_display(result.text)
            print(f"    Display: {display}")
            print(f"    Valid: {is_valid} ({reason})")
        else:
            print(f"    No text to validate")
        
        print(f"\n  {'-'*60}")


def analyze_preprocessing(image_path: str):
    """Analyze preprocessing effects on a sample plate."""
    
    img = cv2.imread(image_path)
    if img is None:
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get first detection
    detections, _ = detect_with_character_validation(gray, min_char_score=0.3)
    if not detections:
        return
    
    x, y, w, h = detections[0]['box']
    roi = gray[y:y+h, x:x+w]
    
    print(f"\n{'='*70}")
    print("PREPROCESSING ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nOriginal ROI size: {roi.shape}")
    
    # Test different preprocessing
    tests = [
        ("Raw", roi),
        ("Resized 50px", resize_for_ocr(roi, 50)),
        ("Resized 60px", resize_for_ocr(roi, 60)),
        ("Resized 80px", resize_for_ocr(roi, 80)),
    ]
    
    configure_tesseract()
    
    for name, img in tests:
        pre = binarize_for_ocr(img)
        pre = pad_image(pre, 12)
        
        result = ocr_plate_multi_psm(img, vn_plate=True)
        print(f"\n{name}:")
        print(f"  Size: {pre.shape}")
        print(f"  Result: '{result.text}' (conf: {result.mean_conf:.1f}%)")


def main():
    """Run detailed analysis."""
    
    if not os.path.exists(TEST_FOLDER):
        print(f"Test folder not found: {TEST_FOLDER}")
        return
    
    # Get test images
    images = sorted([f for f in os.listdir(TEST_FOLDER) if f.lower().endswith('.jpg')])[:3]
    
    for img_name in images:
        img_path = os.path.join(TEST_FOLDER, img_name)
        test_single_image_detailed(img_path)
    
    # Preprocessing analysis on first image
    if images:
        img_path = os.path.join(TEST_FOLDER, images[0])
        analyze_preprocessing(img_path)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
