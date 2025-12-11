#!/usr/bin/env python3
"""
Debug script để phân tích tại sao detection thất bại với ảnh có nhiều contours.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.lp_detector import (
    detect_multi_preset,
    detect_with_character_validation,
    detect_with_edge_backup,
    filter_detections_by_characters,
    PRESETS,
    auto_canny,
)
from src.pipeline import LicensePlateRecognizer


def analyze_image(image_path: str):
    """Phân tích chi tiết quá trình detection."""
    
    print("=" * 70)
    print(f"PHÂN TÍCH ẢNH: {image_path}")
    print("=" * 70)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Không thể load ảnh {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    img_area = h * w
    
    print(f"\n📷 Kích thước ảnh: {w}x{h} = {img_area:,} pixels")
    
    # ========================================
    # Step 1: Analyze Canny edge detection
    # ========================================
    print("\n" + "=" * 50)
    print("1️⃣ CANNY EDGE DETECTION")
    print("=" * 50)
    
    lower, upper = auto_canny(gray, sigma=0.33)
    edges = cv2.Canny(gray, lower, upper)
    edge_pixels = np.count_nonzero(edges)
    edge_ratio = edge_pixels / img_area * 100
    
    print(f"  Ngưỡng Canny: {lower} - {upper}")
    print(f"  Số edge pixels: {edge_pixels:,} ({edge_ratio:.2f}% ảnh)")
    
    if edge_ratio > 10:
        print(f"  ⚠️ CẢNH BÁO: Quá nhiều edge ({edge_ratio:.2f}% > 10%)")
        print(f"     Ảnh có nhiều chi tiết, texture sẽ tạo nhiều contours")
    
    # ========================================
    # Step 2: Analyze contours per preset
    # ========================================
    print("\n" + "=" * 50)
    print("2️⃣ CONTOUR DETECTION THEO PRESET")
    print("=" * 50)
    
    for preset_name in ["car_square", "car_rect", "bike"]:
        cfg = PRESETS[preset_name]
        
        # Morphology
        kernel_close = cfg.kernel_close or (25, 5)
        kernel_open = cfg.kernel_open or (5, 5)
        
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, 
                                  cv2.getStructuringElement(cv2.MORPH_RECT, kernel_close))
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, kernel_open))
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"\n  📦 Preset: {preset_name}")
        print(f"     Kernel Close: {kernel_close}, Open: {kernel_open}")
        print(f"     Số contours: {len(contours)}")
        
        # Filter contours
        min_area = img_area * cfg.min_area_ratio
        max_area = img_area * cfg.max_area_ratio
        valid_count = 0
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            x, y, cw, ch = cv2.boundingRect(cnt)
            ar = cw / ch if ch > 0 else 0
            
            if cfg.aspect_ratio_range[0] <= ar <= cfg.aspect_ratio_range[1]:
                valid_count += 1
                candidates.append({
                    "box": (x, y, cw, ch),
                    "ar": ar,
                    "area": area
                })
        
        print(f"     Sau filter (area + AR): {valid_count} candidates")
        print(f"     AR range: {cfg.aspect_ratio_range}")
        print(f"     Area range: {min_area:.0f} - {max_area:.0f}")
        
        if candidates:
            print(f"     Candidates:")
            for c in candidates[:5]:  # Show top 5
                print(f"       - Box: {c['box']}, AR: {c['ar']:.2f}")
    
    # ========================================
    # Step 3: Full detection pipeline
    # ========================================
    print("\n" + "=" * 50)
    print("3️⃣ FULL DETECTION PIPELINE")
    print("=" * 50)
    
    detections, debug_info = detect_with_character_validation(
        gray,
        bgr_image=image,
        min_char_score=0.35,
        debug=True
    )
    
    print(f"\n  Candidates từ detection: {debug_info.get('candidates_count', 0)}")
    print(f"  Valid sau character validation: {debug_info.get('valid_count', 0)}")
    
    # Xem chi tiết character validation
    char_stats = debug_info.get('char_validation', {})
    if 'details' in char_stats:
        print(f"\n  Chi tiết validation:")
        for det in char_stats['details'][:10]:  # Top 10
            box = det.get('box', 'N/A')
            score = det.get('score', 0)
            chars = det.get('char_count', 0)
            valid = det.get('valid', False)
            reason = det.get('reason', '')
            print(f"    Box={box}, Score={score:.2f}, Chars={chars}, Valid={valid}")
            if reason:
                print(f"      Reason: {reason}")
    
    # ========================================
    # Step 4: Full recognizer
    # ========================================
    print("\n" + "=" * 50)
    print("4️⃣ FULL RECOGNIZER (EasyOCR)")
    print("=" * 50)
    
    recognizer = LicensePlateRecognizer(
        ocr_engine="easyocr",
        use_character_validation=True,
        debug=True
    )
    
    result = recognizer.recognize(image)
    
    print(f"\n  Số biển số phát hiện: {len(result.plates)}")
    
    if result.plates:
        for i, plate in enumerate(result.plates):
            print(f"\n  Biển số {i+1}:")
            print(f"    Text: {plate.text}")
            print(f"    Confidence: {plate.confidence:.1f}%")
            print(f"    Box: {plate.box}")
            print(f"    Method: {plate.detection_method}")
    else:
        print("\n  ❌ KHÔNG PHÁT HIỆN ĐƯỢC BIỂN SỐ!")
        print("\n  Nguyên nhân có thể:")
        print("    1. Quá nhiều contours gây nhiễu")
        print("    2. Character validation score quá thấp")
        print("    3. Biển số bị lẫn với các vùng khác")
        
        # Thử hard mode
        print("\n  Thử với Hard Mode (Color + MSER)...")
        recognizer_hard = LicensePlateRecognizer(
            ocr_engine="easyocr",
            use_character_validation=True,
            use_color_detection=True,
            use_mser_detection=True,
            debug=True
        )
        result_hard = recognizer_hard.recognize(image)
        print(f"  Kết quả Hard Mode: {len(result_hard.plates)} plates")
        if result_hard.plates:
            for plate in result_hard.plates:
                print(f"    - {plate.text} ({plate.confidence:.1f}%)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Path to image")
    args = parser.parse_args()
    
    analyze_image(args.image)
