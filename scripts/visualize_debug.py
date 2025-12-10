# -*- coding: utf-8 -*-
"""
Visual Debug Tool - Visualize detection and OCR results
Save debug images for analysis
"""

import sys
sys.path.insert(0, r'F:\CODE\XLA')

import cv2
import numpy as np
from pathlib import Path

from src.lp_detector import detect_with_character_validation, detect_multi_preset
from src.character_segmenter import segment_characters_multi_method, binarize_plate
from src.ocr_engine import ocr_single_char, resize_for_ocr, ocr_plate_multi_psm
from src.utils import binarize_for_ocr

def visualize_single_image(img_path: str, output_dir: str = "debug_output"):
    """Visualize detection and segmentation for a single image"""
    
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    base_name = Path(img_path).stem
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Image: {base_name}")
    print(f"Size: {img.shape[1]}x{img.shape[0]}")
    
    # Save original with detections
    vis_img = img.copy()
    
    # Detect plates - returns (list of boxes, debug_info)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections, debug_info = detect_multi_preset(gray, debug=False)
    
    print(f"Detections: {len(detections)}")
    
    for i, box in enumerate(detections):
        x, y, w, h = box
        aspect_ratio = w / h if h > 0 else 0
        
        print(f"\n  Detection {i+1}:")
        print(f"    Box: ({x}, {y}, {w}, {h})")
        print(f"    Aspect ratio: {aspect_ratio:.2f}")
        
        # Color code by type
        if aspect_ratio >= 2.5:
            color = (0, 255, 0)  # Green - 1-line plate
            plate_type = "1-line"
        elif aspect_ratio >= 1.2:
            color = (255, 255, 0)  # Cyan - wide 2-line
            plate_type = "wide-2line"
        else:
            color = (0, 0, 255)  # Red - square (2-line)
            plate_type = "square"
        
        # Draw box on visualization
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis_img, f"#{i+1} AR:{aspect_ratio:.1f}", 
                   (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Extract ROI
        roi = img[y:y+h, x:x+w]
        
        # Save ROI
        roi_path = out_dir / f"{base_name}_det{i+1}_roi.jpg"
        cv2.imwrite(str(roi_path), roi)
        print(f"    Saved ROI: {roi_path.name}")
        
        # Binarize ROI - returns (binary, inverted)
        binary, binary_inv = binarize_plate(roi)
        binary_path = out_dir / f"{base_name}_det{i+1}_binary.jpg"
        cv2.imwrite(str(binary_path), binary)
        
        # Segment characters - returns SegmentationResult
        seg_result = segment_characters_multi_method(roi)
        chars = seg_result.char_images
        print(f"    Characters found: {len(chars)}")
        
        # Visualize segmented characters
        if chars:
            # Create character strip
            char_strip = []
            for j, char_img in enumerate(chars):
                # Resize for visualization
                h_char, w_char = char_img.shape[:2]
                scale = 50 / max(h_char, 1)
                new_w = max(int(w_char * scale), 20)
                new_h = 50
                resized = cv2.resize(char_img, (new_w, new_h))
                
                # Convert to BGR if grayscale
                if len(resized.shape) == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                
                # Add border
                bordered = cv2.copyMakeBorder(resized, 2, 2, 2, 2, 
                                              cv2.BORDER_CONSTANT, value=(0,255,0))
                char_strip.append(bordered)
            
            # Concatenate horizontally
            strip = np.hstack(char_strip)
            strip_path = out_dir / f"{base_name}_det{i+1}_chars.jpg"
            cv2.imwrite(str(strip_path), strip)
            print(f"    Saved char strip: {strip_path.name}")
            
            # OCR each character
            ocr_results = []
            for j, char_img in enumerate(chars):
                # Prepare for OCR
                prepared = resize_for_ocr(char_img, target_height=40)
                prepared_bin = binarize_for_ocr(prepared, sharpen=True)
                
                # OCR
                char_text, conf = ocr_single_char(prepared_bin)
                ocr_results.append((char_text, conf))
                print(f"      Char {j+1}: '{char_text}' ({conf:.0f}%)")
            
            # Line OCR - returns OCRResult
            ocr_result = ocr_plate_multi_psm(roi)
            print(f"    Line OCR: '{ocr_result.text}' ({ocr_result.mean_conf:.0f}%)")
    
    # Save visualization
    vis_path = out_dir / f"{base_name}_detections.jpg"
    cv2.imwrite(str(vis_path), vis_img)
    print(f"\n  Saved visualization: {vis_path.name}")


def main():
    # Test with a few sample images
    test_dir = Path(r"F:\CODE\XLA\data\test_images\CarTGMT")
    output_dir = Path(r"F:\CODE\XLA\debug_output")
    output_dir.mkdir(exist_ok=True)
    
    # Select a few diverse images
    sample_files = [
        # AEON samples - seem to have different format
        "AEONTP_5026788_checkin_2020-1-13-16-19_f7zXETONv.jpg",
        "AEONTP_51F86947_checkin_2020-1-13-16-15sUVxP1Ihlt.jpg",
        # AQUA samples with different plate numbers
        "AQUA2_09566_checkin_2020-10-22-10-45NGiXXm1L2u.jpg",
        "AQUA4_01418_checkin_2020-10-22-13-28x1xGZQit5B.jpg", 
        "AQUA5_09566_checkin_2020-10-22-8-18mr16q4sxtm.jpg",
        "AQUA7_51624_checkin_2020-11-1-9-21B5RmLxA6l3.jpg",
    ]
    
    for fname in sample_files:
        img_path = test_dir / fname
        if img_path.exists():
            visualize_single_image(str(img_path), str(output_dir))
        else:
            print(f"File not found: {fname}")


if __name__ == "__main__":
    main()
