"""
OCR Performance Analysis Script
===============================

This script analyzes the current OCR performance to identify bottlenecks
and areas for improvement.

Key metrics:
- Detection accuracy (are we finding the plates?)
- Segmentation quality (are characters properly separated?)
- OCR confidence (how confident is Tesseract?)
- Format validation (do results match VN plate patterns?)
"""
import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import LicensePlateRecognizer
from src.lp_detector import detect_with_character_validation
from src.character_segmenter import segment_characters, binarize_plate
from src.ocr_engine import (
    validate_vn_plate_format, 
    format_plate_display,
    ocr_single_char,
    ocr_plate_line,
    configure_tesseract,
)


TEST_FOLDER = r'F:\CODE\XLA\data\test_images\CarTGMT'


def analyze_detection_quality(test_folder: str, num_images: int = 20):
    """Analyze detection stage performance."""
    print("\n" + "="*60)
    print("1. DETECTION QUALITY ANALYSIS")
    print("="*60)
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])[:num_images]
    
    total_detections = 0
    high_char_score = 0
    detection_methods = defaultdict(int)
    aspect_ratios = []
    
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections, info = detect_with_character_validation(gray, debug=True)
        
        for det in detections:
            total_detections += 1
            if det.get('char_score', 0) >= 0.5:
                high_char_score += 1
            
            method = det.get('method', 'unknown')
            detection_methods[method] += 1
            
            x, y, w, h = det['box']
            ar = w / float(h) if h > 0 else 1
            aspect_ratios.append(ar)
    
    print(f"\nImages analyzed: {len(images)}")
    print(f"Total detections: {total_detections}")
    print(f"High char score (>=0.5): {high_char_score} ({100*high_char_score/max(1,total_detections):.1f}%)")
    
    print(f"\nDetection methods distribution:")
    for method, count in sorted(detection_methods.items()):
        print(f"  {method}: {count} ({100*count/max(1,total_detections):.1f}%)")
    
    if aspect_ratios:
        print(f"\nAspect ratio statistics:")
        print(f"  Mean: {np.mean(aspect_ratios):.2f}")
        print(f"  Std: {np.std(aspect_ratios):.2f}")
        print(f"  Min: {np.min(aspect_ratios):.2f}")
        print(f"  Max: {np.max(aspect_ratios):.2f}")
    
    return total_detections, high_char_score


def analyze_segmentation_quality(test_folder: str, num_images: int = 10):
    """Analyze character segmentation quality."""
    print("\n" + "="*60)
    print("2. SEGMENTATION QUALITY ANALYSIS")
    print("="*60)
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])[:num_images]
    
    char_counts = []
    good_segmentations = 0
    total_rois = 0
    
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections, _ = detect_with_character_validation(gray, min_char_score=0.3)
        
        for det in detections[:3]:  # Top 3 detections per image
            x, y, w, h = det['box']
            roi = gray[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
            
            total_rois += 1
            result = segment_characters(roi, plate_type='car2', debug=True)
            num_chars = len(result.boxes)
            char_counts.append(num_chars)
            
            # Good segmentation: 5-9 characters (typical VN plate)
            if 5 <= num_chars <= 9:
                good_segmentations += 1
    
    print(f"\nROIs analyzed: {total_rois}")
    print(f"Good segmentations (5-9 chars): {good_segmentations} ({100*good_segmentations/max(1,total_rois):.1f}%)")
    
    if char_counts:
        print(f"\nCharacter count statistics:")
        print(f"  Mean: {np.mean(char_counts):.1f}")
        print(f"  Std: {np.std(char_counts):.1f}")
        print(f"  Distribution:")
        for count in range(0, 15):
            num = char_counts.count(count)
            if num > 0:
                print(f"    {count} chars: {num} ({100*num/len(char_counts):.1f}%)")
    
    return good_segmentations, total_rois


def analyze_ocr_confidence(test_folder: str, num_images: int = 10):
    """Analyze OCR confidence levels."""
    print("\n" + "="*60)
    print("3. OCR CONFIDENCE ANALYSIS")
    print("="*60)
    
    configure_tesseract()
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])[:num_images]
    
    all_confidences = []
    high_conf_count = 0
    valid_format_count = 0
    total_plates = 0
    
    recognizer = LicensePlateRecognizer(debug=True)
    
    results_detail = []
    
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        result = recognizer.recognize(img)
        
        for plate in result.plates[:3]:
            total_plates += 1
            conf = plate.confidence
            all_confidences.append(conf)
            
            if conf >= 90:
                high_conf_count += 1
            
            is_valid, _ = validate_vn_plate_format(plate.text) if plate.text else (False, "")
            if is_valid:
                valid_format_count += 1
            
            results_detail.append({
                'image': img_name[:25],
                'text': plate.text,
                'confidence': conf,
                'valid': is_valid,
                'type': plate.plate_type,
            })
    
    print(f"\nTotal plates OCR'd: {total_plates}")
    print(f"High confidence (>=90%): {high_conf_count} ({100*high_conf_count/max(1,total_plates):.1f}%)")
    print(f"Valid VN format: {valid_format_count} ({100*valid_format_count/max(1,total_plates):.1f}%)")
    
    if all_confidences:
        print(f"\nConfidence statistics:")
        print(f"  Mean: {np.mean(all_confidences):.1f}%")
        print(f"  Std: {np.std(all_confidences):.1f}%")
        print(f"  Min: {np.min(all_confidences):.1f}%")
        print(f"  Max: {np.max(all_confidences):.1f}%")
        
        # Distribution
        print(f"\nConfidence distribution:")
        ranges = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
        for low, high in ranges:
            count = sum(1 for c in all_confidences if low <= c < high)
            print(f"  {low}-{high}%: {count} ({100*count/len(all_confidences):.1f}%)")
    
    # Show some examples
    print(f"\nSample results (sorted by confidence):")
    print("-" * 70)
    sorted_results = sorted(results_detail, key=lambda x: x['confidence'], reverse=True)
    for r in sorted_results[:15]:
        display = format_plate_display(r['text']) if r['text'] else 'N/A'
        valid_mark = 'Y' if r['valid'] else 'N'
        print(f"  {r['image']:<25} {display:<15} {r['confidence']:>5.1f}% {valid_mark}")
    
    return high_conf_count, total_plates, valid_format_count


def analyze_tesseract_settings(test_folder: str):
    """Test different Tesseract settings."""
    print("\n" + "="*60)
    print("4. TESSERACT SETTINGS COMPARISON")
    print("="*60)
    
    configure_tesseract()
    
    # Get a few plate ROIs
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])[:5]
    
    rois = []
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections, _ = detect_with_character_validation(gray, min_char_score=0.4)
        
        for det in detections[:1]:
            x, y, w, h = det['box']
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
                rois.append((img_name[:20], roi))
    
    print(f"\nTesting {len(rois)} plate ROIs with different PSM modes:")
    print("-" * 70)
    
    # Test different PSM modes
    psm_modes = [6, 7, 8, 11, 13]
    
    for roi_name, roi in rois[:3]:
        print(f"\n{roi_name}:")
        for psm in psm_modes:
            try:
                result = ocr_plate_line(roi, vn_plate=True, psm=psm)
                print(f"  PSM {psm:2d}: {result.text:<15} (conf: {result.mean_conf:.1f}%)")
            except Exception as e:
                print(f"  PSM {psm:2d}: ERROR - {e}")


def suggest_improvements():
    """Suggest parameter improvements based on analysis."""
    print("\n" + "="*60)
    print("5. SUGGESTED IMPROVEMENTS")
    print("="*60)
    
    suggestions = """
    Based on typical VN plate recognition challenges:

    A. DETECTION IMPROVEMENTS:
    1. Lower min_char_score threshold from 0.35 to 0.25-0.30
       - Catches more true positives at cost of some false positives
    
    2. Adjust aspect ratio ranges:
       - car_square: 0.8-2.0 (currently 0.7-2.0)
       - car_rect: 2.5-6.0 (currently 2.8-6.5)
       - bike: 1.0-3.0 (currently 1.2-3.0)
    
    B. SEGMENTATION IMPROVEMENTS:
    1. Use 'combined' binarization method by default
    2. Increase CLAHE clipLimit from 2.0 to 3.0 for better contrast
    3. Adjust character height filter from 0.15 to 0.12 of ROI height
    
    C. OCR IMPROVEMENTS:
    1. Use PSM 7 (single line) for 1-line plates
    2. Use PSM 6 (block) for 2-line plates  
    3. Increase image padding from 8 to 12 pixels
    4. Apply additional preprocessing:
       - Sharpen before OCR
       - Resize to optimal height (40-50px per line)
    5. Try multiple PSM modes and pick best confidence
    
    D. POST-PROCESSING:
    1. Validate against province codes
    2. Use character position patterns (NN-C-NNNNN)
    3. Correct common OCR errors based on position
    """
    print(suggestions)


def main():
    """Run full analysis."""
    print("\n" + "="*60)
    print("OCR PERFORMANCE ANALYSIS")
    print("="*60)
    
    if not os.path.exists(TEST_FOLDER):
        print(f"❌ Test folder not found: {TEST_FOLDER}")
        return
    
    # Run analyses
    analyze_detection_quality(TEST_FOLDER, num_images=15)
    analyze_segmentation_quality(TEST_FOLDER, num_images=10)
    analyze_ocr_confidence(TEST_FOLDER, num_images=10)
    analyze_tesseract_settings(TEST_FOLDER)
    suggest_improvements()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
