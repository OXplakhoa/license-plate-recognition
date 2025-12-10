# -*- coding: utf-8 -*-
"""Test OCR on detected character images"""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from pathlib import Path

from src.lp_detector import detect_multi_preset
from src.character_segmenter import segment_characters_multi_method
from src.ocr_engine import ocr_single_char, resize_for_ocr, ocr_characters
from src.utils import binarize_for_ocr

def test_ocr_on_image(img_path: str, output_dir: str = "debug_output"):
    """Test OCR on a single image"""
    out_dir = Path(output_dir)
    base_name = Path(img_path).stem[:30]
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect
    detections, _ = detect_multi_preset(gray)
    
    if not detections:
        print(f"{base_name}: No detections")
        return
    
    # Process first detection
    x, y, w, h = detections[0]
    roi = img[y:y+h, x:x+w]
    
    print(f"{base_name}: ROI={w}x{h}")
    
    # Segment
    seg_result = segment_characters_multi_method(roi)
    chars = seg_result.char_images
    
    print(f"  Segmented {len(chars)} characters")
    
    if not chars:
        return
    
    # Save each character and its OCR result
    for i, char_img in enumerate(chars):
        h_c, w_c = char_img.shape[:2]
        
        # Save original char
        char_path = out_dir / f"{base_name}_char{i}_orig.png"
        cv2.imwrite(str(char_path), char_img)
        
        # Process for OCR
        resized = resize_for_ocr(char_img, target_height=40)
        binary = binarize_for_ocr(resized, sharpen=True)
        
        # Save processed
        bin_path = out_dir / f"{base_name}_char{i}_bin.png"
        cv2.imwrite(str(bin_path), binary)
        
        # OCR
        char_text, conf = ocr_single_char(char_img)
        print(f"  Char {i}: size={w_c}x{h_c}, OCR='{char_text}' ({conf:.0f}%)")
    
    # Full OCR result
    result = ocr_characters(chars, vn_plate=True)
    print(f"  Combined: '{result.text}' (mean: {result.mean_conf:.1f}%)")


def main():
    test_dir = Path(r"F:\CODE\XLA\data\test_images\CarTGMT")
    out_dir = Path(r"F:\CODE\XLA\debug_output")
    out_dir.mkdir(exist_ok=True)
    
    # Test on a few images
    test_files = [
        "AEONTP_51F86947_checkin_2020-1-13-16-15sUVxP1Ihlt.jpg",
        "AQUA2_09566_checkin_2020-10-22-10-45NGiXXm1L2u.jpg",
        "AQUA4_01418_checkin_2020-10-22-13-28x1xGZQit5B.jpg",
    ]
    
    for fname in test_files:
        img_path = test_dir / fname
        if img_path.exists():
            test_ocr_on_image(str(img_path), str(out_dir))
            print()


if __name__ == "__main__":
    main()
