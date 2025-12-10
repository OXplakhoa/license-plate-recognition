#!/usr/bin/env python
"""
Detailed Pipeline Visualization - Vietnamese License Plate Recognition
======================================================================
Hiển thị chi tiết từng bước xử lý theo project.txt:
  2. Tiền xử lý ảnh (Preprocessing)
  3. Phát hiện vùng biển số (Detection)
  4. Chuẩn hóa và phân tách ký tự (Segmentation)
  5. Nhận dạng ký tự (OCR)
  6. Hiển thị và đánh giá kết quả
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

from src.lp_detector import (
    detect_with_character_validation,
    correct_plate_perspective_and_skew,
    detect_white_plate_regions,
    detect_with_mser,
    get_plate_type_from_aspect,
)
from src.character_segmenter import segment_characters, binarize_plate
from src.utils import ensure_grayscale
from src.heuristics import apply_heuristics


def create_detailed_visualization(image_path: str, output_path: str = None, ocr_engine: str = "easyocr"):
    """Create detailed step-by-step visualization."""
    
    print("=" * 70)
    print("VISUALIZATION CHI TIẾT - NHẬN DẠNG BIỂN SỐ XE VIỆT NAM")
    print("=" * 70)
    
    # =========================================================================
    # BƯỚC 1: Nhập ảnh đầu vào
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 1: NHẬP ẢNH ĐẦU VÀO")
    print("=" * 70)
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"✓ Đọc ảnh thành công: {image_path}")
    print(f"  - Kích thước: {w_orig}x{h_orig} pixels")
    print(f"  - Số kênh màu: {img_bgr.shape[2]} (BGR)")
    
    # =========================================================================
    # BƯỚC 2: Tiền xử lý ảnh (Preprocessing)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 2: TIỀN XỬ LÝ ẢNH (PREPROCESSING)")
    print("=" * 70)
    
    # 2.1 Chuyển ảnh sang thang xám
    print("\n📍 Bước 2.1: Chuyển sang ảnh xám (Grayscale)")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"  → Từ 3 kênh (BGR) → 1 kênh (Gray)")
    print(f"  → Kích thước dữ liệu: {img_bgr.nbytes} → {gray.nbytes} bytes")
    
    # 2.2 Lọc nhiễu - Gaussian Blur
    print("\n📍 Bước 2.2: Lọc nhiễu - Gaussian Blur")
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    print(f"  → Kernel size: 5x5")
    print(f"  → Sigma: auto (0)")
    
    # 2.3 Lọc nhiễu - Median Filter (alternative)
    print("\n📍 Bước 2.3: Lọc nhiễu - Median Filter")
    median_blur = cv2.medianBlur(gray, 5)
    print(f"  → Kernel size: 5")
    
    # 2.4 Cân bằng độ tương phản - Histogram Equalization
    print("\n📍 Bước 2.4: Cân bằng histogram (Histogram Equalization)")
    hist_eq = cv2.equalizeHist(gray)
    print(f"  → Cân bằng toàn cục histogram")
    
    # 2.5 Cân bằng độ tương phản - CLAHE
    print("\n📍 Bước 2.5: CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    print(f"  → clipLimit: 2.0")
    print(f"  → tileGridSize: 8x8")
    
    # =========================================================================
    # BƯỚC 3: Phát hiện vùng biển số (Detection)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 3: PHÁT HIỆN VÙNG BIỂN SỐ (DETECTION)")
    print("=" * 70)
    
    # 3.1 Phát hiện biên - Canny
    print("\n📍 Bước 3.1: Phát hiện biên - Canny Edge Detection")
    edges_canny = cv2.Canny(gaussian_blur, 50, 150)
    print(f"  → Low threshold: 50")
    print(f"  → High threshold: 150")
    
    # 3.2 Phát hiện biên - Sobel
    print("\n📍 Bước 3.2: Phát hiện biên - Sobel")
    sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = np.uint8(255 * sobel_mag / sobel_mag.max())
    print(f"  → Sobel X (gradient ngang)")
    print(f"  → Sobel Y (gradient dọc)")
    print(f"  → Combined magnitude")
    
    # 3.3 Morphology - Đóng (Closing)
    print("\n📍 Bước 3.3: Phép đóng Morphology (Closing)")
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph_close = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel_close)
    print(f"  → Kernel: 17x3 (RECT)")
    print(f"  → Nối các cạnh gần nhau")
    
    # 3.4 Morphology - Mở (Opening)
    print("\n📍 Bước 3.4: Phép mở Morphology (Opening)")
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel_open)
    print(f"  → Kernel: 3x3 (RECT)")
    print(f"  → Loại bỏ nhiễu nhỏ")
    
    # 3.5 Tìm contour và lọc theo tỷ lệ
    print("\n📍 Bước 3.5: Tìm và lọc Contour")
    contours, _ = cv2.findContours(morph_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  → Tổng số contour tìm được: {len(contours)}")
    
    # Filter contours by aspect ratio
    plate_candidates = []
    img_contours = img_rgb.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / h if h > 0 else 0
        area = w * h
        # Vietnamese plate AR: ~1.5-5.0
        if 1.0 < ar < 6.0 and area > 1000:
            plate_candidates.append((x, y, w, h, ar))
            cv2.rectangle(img_contours, (x, y), (x+w, y+h), (255, 165, 0), 2)
    
    print(f"  → Contour phù hợp tỷ lệ biển số: {len(plate_candidates)}")
    
    # 3.6 Detection với Character Validation (với auto-retry như pipeline chính)
    print("\n📍 Bước 3.6: Detection với Character Validation")
    
    # First try: Standard detection
    detections, det_info = detect_with_character_validation(
        gray, 
        bgr_image=img_bgr,
        min_char_score=0.35, 
        debug=True
    )
    valid_detections = [d for d in detections if d.get("char_valid", False)]
    detection_method = "contour+edge"
    
    # Luôn thử thêm color + MSER để có nhiều candidates hơn
    print("  → Thử thêm Color + MSER detection...")
    color_detections, color_info = detect_with_character_validation(
        gray,
        bgr_image=img_bgr,
        min_char_score=0.25,  # Lower threshold
        use_color=True,
        use_mser=True,
        debug=True
    )
    
    # Merge detections, prioritize by char_score
    all_detections = detections + [d for d in color_detections if d.get("char_valid", False)]
    
    # Remove duplicates (same box)
    seen_boxes = set()
    unique_detections = []
    for d in all_detections:
        box_key = tuple(d['box'])
        if box_key not in seen_boxes:
            seen_boxes.add(box_key)
            unique_detections.append(d)
    
    # Sort by char_score
    unique_detections.sort(key=lambda d: d.get('char_score', 0), reverse=True)
    detections = unique_detections
    
    # Determine which method found the best detection
    if detections:
        best_source = detections[0].get('source', detections[0].get('method', 'unknown'))
        if best_source in ['color', 'mser']:
            detection_method = f"color+mser ({best_source})"
        else:
            detection_method = "contour+edge"
    
    print(f"  → Số vùng phát hiện: {len(detections)}")
    print(f"  → Phương pháp tốt nhất: {detection_method}")
    
    if detections:
        best_det = detections[0]
        x, y, w, h = best_det['box']
        source = best_det.get('source', best_det.get('method', 'unknown'))
        print(f"  → Vùng tốt nhất: ({x}, {y}) - {w}x{h}")
        print(f"  → Điểm char_score: {best_det.get('char_score', 0):.2f}")
        print(f"  → Nguồn: {source}")
    
    # Extract ROI
    roi_gray = None
    plate_type = "car2"
    if detections:
        x, y, w, h = detections[0]['box']
        roi_gray = gray[y:y+h, x:x+w]
        ar = w / h
        area = w * h
        # Use improved classification with size
        plate_type = get_plate_type_from_aspect(ar, w, h)
        print(f"  → Loại biển số: {plate_type} (AR={ar:.2f}, Size={w}x{h}, Area={area})")
    
    # =========================================================================
    # BƯỚC 4: Chuẩn hóa và phân tách ký tự (Segmentation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 4: CHUẨN HÓA VÀ PHÂN TÁCH KÝ TỰ (SEGMENTATION)")
    print("=" * 70)
    
    if roi_gray is not None:
        # 4.0 Perspective Correction
        print("\n📍 Bước 4.0: Hiệu chỉnh phối cảnh (Perspective Correction)")
        roi_box = (0, 0, roi_gray.shape[1], roi_gray.shape[0])
        roi_corrected, correction_info = correct_plate_perspective_and_skew(roi_gray, roi=roi_box, deskew=True)
        if roi_corrected is None:
            roi_corrected = roi_gray
            print(f"  → Không cần hiệu chỉnh")
        else:
            print(f"  → Đã hiệu chỉnh phối cảnh")
            if correction_info and 'skew_angle' in correction_info:
                print(f"  → Góc nghiêng: {correction_info['skew_angle']:.1f}°")
        
        # 4.1 Chuyển sang nhị phân - Otsu Threshold
        print("\n📍 Bước 4.1: Nhị phân hóa - Otsu Threshold")
        _, otsu_binary = cv2.threshold(roi_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_inv = cv2.bitwise_not(otsu_binary)
        print(f"  → Phương pháp: Otsu's automatic thresholding")
        print(f"  → Tự động tìm ngưỡng tối ưu")
        
        # 4.2 Adaptive Threshold (alternative)
        print("\n📍 Bước 4.2: Nhị phân hóa - Adaptive Threshold")
        adaptive_binary = cv2.adaptiveThreshold(
            roi_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5
        )
        print(f"  → Block size: 15")
        print(f"  → C constant: 5")
        
        # 4.3 Combined method (best of both)
        print("\n📍 Bước 4.3: Phương pháp kết hợp (Combined)")
        binary_combined, inv_combined = binarize_plate(roi_corrected, method='combined', denoise=False)
        print(f"  → Kết hợp Otsu + Adaptive")
        print(f"  → Chọn phương pháp có nhiều ký tự nhất")
        
        # 4.4 Tìm contour ký tự
        print("\n📍 Bước 4.4: Tìm Contour ký tự")
        contours_char, _ = cv2.findContours(cv2.bitwise_not(binary_combined), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  → Tổng contour: {len(contours_char)}")
        
        # Filter character contours
        char_contours = []
        roi_h, roi_w = roi_corrected.shape[:2]
        for cnt in contours_char:
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / h if h > 0 else 0
            hr = h / roi_h
            area = w * h
            if 0.1 < ar < 1.5 and 0.15 < hr < 0.6 and area > 50:
                char_contours.append((x, y, w, h))
        
        print(f"  → Contour ký tự hợp lệ: {len(char_contours)}")
        
        # 4.5 Phân đoạn ký tự
        print("\n📍 Bước 4.5: Phân đoạn ký tự (Character Segmentation)")
        seg_result = segment_characters(roi_corrected, plate_type=plate_type, debug=True)
        print(f"  → Số ký tự phát hiện: {len(seg_result.boxes)}")
        print(f"  → Debug: {seg_result.debug_info}")
        
        # 4.6 Căn chỉnh và resize ký tự
        print("\n📍 Bước 4.6: Căn chỉnh và Resize ký tự")
        char_images_28x28 = []
        for i, (cx, cy, cw, ch) in enumerate(seg_result.boxes):
            char_img = seg_result.inverted_binary[cy:cy+ch, cx:cx+cw]
            # Resize to 28x28 with padding
            side = max(cw, ch) + 4
            canvas = np.zeros((side, side), dtype=np.uint8)
            sx = (side - cw) // 2
            sy = (side - ch) // 2
            canvas[sy:sy+ch, sx:sx+cw] = char_img
            resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
            char_images_28x28.append(resized)
        print(f"  → {len(char_images_28x28)} ký tự được resize về 28x28")
    else:
        roi_corrected = None
        otsu_binary = None
        otsu_inv = None
        adaptive_binary = None
        binary_combined = None
        inv_combined = None
        seg_result = None
        char_images_28x28 = []
        char_contours = []
    
    # =========================================================================
    # BƯỚC 5: Nhận dạng ký tự (OCR)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 5: NHẬN DẠNG KÝ TỰ (OCR)")
    print("=" * 70)
    
    ocr_text = ""
    ocr_confidence = 0.0
    char_results = []
    
    if roi_corrected is not None:
        if ocr_engine == "easyocr":
            print("\n📍 Sử dụng EasyOCR (Deep Learning based)")
            try:
                import easyocr
                print("  → Loading EasyOCR model...")
                reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                results = reader.readtext(roi_corrected, detail=1)
                
                texts = []
                confs = []
                for bbox, text, conf in results:
                    texts.append(text)
                    confs.append(conf)
                    char_results.append({'text': text, 'conf': conf, 'bbox': bbox})
                    print(f"  → Detected: '{text}' (conf={conf:.2%})")
                
                ocr_text = ''.join(texts).replace(' ', '').upper()
                ocr_confidence = np.mean(confs) * 100 if confs else 0
                
            except Exception as e:
                print(f"  ❌ EasyOCR error: {e}")
        else:
            print("\n📍 Sử dụng Tesseract OCR")
            try:
                import pytesseract
                
                # Per-character OCR
                print("  → OCR từng ký tự...")
                chars = []
                for i, char_img in enumerate(seg_result.char_images if seg_result else []):
                    text = pytesseract.image_to_string(
                        char_img, 
                        config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    ).strip()
                    chars.append(text[:1] if text else '?')
                    char_results.append({'text': text[:1] if text else '?', 'conf': -1})
                
                ocr_text = ''.join(chars)
                
                # Also try full plate OCR
                print("  → OCR toàn bộ biển số...")
                full_text = pytesseract.image_to_string(
                    roi_corrected,
                    config='--psm 7 --oem 3'
                ).strip().replace(' ', '').upper()
                print(f"  → Full OCR: '{full_text}'")
                
                if len(full_text) > len(ocr_text):
                    ocr_text = full_text
                    
            except Exception as e:
                print(f"  ❌ Tesseract error: {e}")
        
        print(f"\n  🔤 Kết quả OCR: {ocr_text}")
        print(f"  📊 Confidence: {ocr_confidence:.2f}%")
    
    # =========================================================================
    # BƯỚC 6: Hiển thị và đánh giá kết quả
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 6: HIỂN THỊ VÀ ĐÁNH GIÁ KẾT QUẢ")
    print("=" * 70)
    
    # Post-processing - sử dụng heuristics mới (bao gồm province code fix)
    from src.ocr_engine import format_plate_display, validate_vn_plate_format
    
    # Apply heuristics (đã import ở đầu file)
    corrected_text = apply_heuristics(ocr_text)
    formatted_text = format_plate_display(corrected_text)
    is_valid = validate_vn_plate_format(corrected_text)
    
    print(f"\n📍 Hậu xử lý:")
    print(f"  → Text gốc: {ocr_text}")
    print(f"  → Sau heuristics: {corrected_text}")
    print(f"  → Formatted: {formatted_text}")
    print(f"  → Valid format: {'✓ Có' if is_valid else '✗ Không'}")
    
    # =========================================================================
    # TẠO VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("TẠO HÌNH ẢNH VISUALIZATION")
    print("=" * 70)
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(24, 32))
    fig.suptitle('PIPELINE NHẬN DẠNG BIỂN SỐ XE VIỆT NAM - CHI TIẾT TỪNG BƯỚC', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Create grid: 8 rows
    gs = gridspec.GridSpec(8, 6, figure=fig, hspace=0.3, wspace=0.2)
    
    # ----- ROW 1: Input & Grayscale -----
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(img_rgb)
    ax1.set_title('1. Ảnh gốc (Input)', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.imshow(gray, cmap='gray')
    ax2.set_title('2.1 Grayscale', fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax3.imshow(gaussian_blur, cmap='gray')
    ax3.set_title('2.2 Gaussian Blur (5x5)', fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # ----- ROW 2: More Preprocessing -----
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.imshow(median_blur, cmap='gray')
    ax4.set_title('2.3 Median Filter', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax5.imshow(hist_eq, cmap='gray')
    ax5.set_title('2.4 Histogram Equalization', fontsize=11, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 4:6])
    ax6.imshow(clahe_img, cmap='gray')
    ax6.set_title('2.5 CLAHE', fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    # ----- ROW 3: Edge Detection -----
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.imshow(edges_canny, cmap='gray')
    ax7.set_title('3.1 Canny Edges', fontsize=11, fontweight='bold')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.imshow(sobel_mag, cmap='gray')
    ax8.set_title('3.2 Sobel Magnitude', fontsize=11, fontweight='bold')
    ax8.axis('off')
    
    ax9 = fig.add_subplot(gs[2, 4:6])
    ax9.imshow(morph_close, cmap='gray')
    ax9.set_title('3.3 Morphology Close', fontsize=11, fontweight='bold')
    ax9.axis('off')
    
    # ----- ROW 4: Contours & Detection -----
    ax10 = fig.add_subplot(gs[3, 0:2])
    ax10.imshow(morph_open, cmap='gray')
    ax10.set_title('3.4 Morphology Open', fontsize=11, fontweight='bold')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[3, 2:4])
    ax11.imshow(img_contours)
    ax11.set_title(f'3.5 Contours ({len(plate_candidates)} candidates)', fontsize=11, fontweight='bold')
    ax11.axis('off')
    
    # Show detection result
    ax12 = fig.add_subplot(gs[3, 4:6])
    img_detected = img_rgb.copy()
    if detections:
        x, y, w, h = detections[0]['box']
        cv2.rectangle(img_detected, (x, y), (x+w, y+h), (0, 255, 0), 3)
    ax12.imshow(img_detected)
    ax12.set_title('3.6 Detection Result', fontsize=11, fontweight='bold')
    ax12.axis('off')
    
    # ----- ROW 5: ROI & Binarization -----
    if roi_corrected is not None:
        ax13 = fig.add_subplot(gs[4, 0])
        ax13.imshow(roi_gray, cmap='gray')
        ax13.set_title('4.0a ROI gốc', fontsize=10, fontweight='bold')
        ax13.axis('off')
        
        ax14 = fig.add_subplot(gs[4, 1])
        ax14.imshow(roi_corrected, cmap='gray')
        ax14.set_title('4.0b Đã hiệu chỉnh', fontsize=10, fontweight='bold')
        ax14.axis('off')
        
        ax15 = fig.add_subplot(gs[4, 2])
        ax15.imshow(otsu_binary, cmap='gray')
        ax15.set_title('4.1 Otsu Binary', fontsize=10, fontweight='bold')
        ax15.axis('off')
        
        ax16 = fig.add_subplot(gs[4, 3])
        ax16.imshow(otsu_inv, cmap='gray')
        ax16.set_title('4.1b Otsu Inverted', fontsize=10, fontweight='bold')
        ax16.axis('off')
        
        ax17 = fig.add_subplot(gs[4, 4])
        ax17.imshow(adaptive_binary, cmap='gray')
        ax17.set_title('4.2 Adaptive', fontsize=10, fontweight='bold')
        ax17.axis('off')
        
        ax18 = fig.add_subplot(gs[4, 5])
        ax18.imshow(cv2.bitwise_not(binary_combined), cmap='gray')
        ax18.set_title('4.3 Combined', fontsize=10, fontweight='bold')
        ax18.axis('off')
    
    # ----- ROW 6: Character Segmentation -----
    if seg_result is not None and len(seg_result.boxes) > 0:
        ax19 = fig.add_subplot(gs[5, 0:2])
        # Show contours on ROI
        roi_contour_img = cv2.cvtColor(roi_corrected, cv2.COLOR_GRAY2RGB)
        for cx, cy, cw, ch in char_contours:
            cv2.rectangle(roi_contour_img, (cx, cy), (cx+cw, cy+ch), (255, 165, 0), 1)
        ax19.imshow(roi_contour_img)
        ax19.set_title(f'4.4 Contour ký tự ({len(char_contours)})', fontsize=10, fontweight='bold')
        ax19.axis('off')
        
        ax20 = fig.add_subplot(gs[5, 2:4])
        # Show segmented boxes
        roi_seg_img = cv2.cvtColor(roi_corrected, cv2.COLOR_GRAY2RGB)
        for i, (cx, cy, cw, ch) in enumerate(seg_result.boxes):
            cv2.rectangle(roi_seg_img, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)
            cv2.putText(roi_seg_img, str(i+1), (cx, cy-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        ax20.imshow(roi_seg_img)
        ax20.set_title(f'4.5 Segmentation ({len(seg_result.boxes)} chars)', fontsize=10, fontweight='bold')
        ax20.axis('off')
        
        # Show individual characters
        ax21 = fig.add_subplot(gs[5, 4:6])
        if char_images_28x28:
            n_chars = len(char_images_28x28)
            cols = min(n_chars, 10)
            rows = (n_chars + cols - 1) // cols
            char_grid = np.zeros((rows * 32, cols * 32), dtype=np.uint8)
            for i, char_img in enumerate(char_images_28x28):
                r, c = i // cols, i % cols
                char_grid[r*32+2:r*32+30, c*32+2:c*32+30] = char_img
            ax21.imshow(char_grid, cmap='gray')
        ax21.set_title('4.6 Ký tự 28x28', fontsize=10, fontweight='bold')
        ax21.axis('off')
    
    # ----- ROW 7: OCR Process -----
    ax22 = fig.add_subplot(gs[6, 0:3])
    if roi_corrected is not None:
        ax22.imshow(roi_corrected, cmap='gray')
        # Add OCR results as text overlay
        ax22.text(0.5, -0.15, f'OCR Input: ROI đã xử lý', transform=ax22.transAxes, 
                  ha='center', fontsize=10)
    ax22.set_title(f'5. OCR ({ocr_engine.upper()})', fontsize=11, fontweight='bold')
    ax22.axis('off')
    
    ax23 = fig.add_subplot(gs[6, 3:6])
    ax23.text(0.5, 0.7, f'Kết quả OCR:', ha='center', va='center', fontsize=14, fontweight='bold',
              transform=ax23.transAxes)
    ax23.text(0.5, 0.5, f'"{ocr_text}"', ha='center', va='center', fontsize=20, fontweight='bold',
              color='blue', transform=ax23.transAxes)
    ax23.text(0.5, 0.3, f'Confidence: {ocr_confidence:.1f}%', ha='center', va='center', fontsize=12,
              transform=ax23.transAxes)
    ax23.axis('off')
    ax23.set_title('5. Kết quả OCR', fontsize=11, fontweight='bold')
    
    # ----- ROW 8: Final Result -----
    ax24 = fig.add_subplot(gs[7, 0:2])
    ax24.imshow(img_rgb)
    if detections:
        x, y, w, h = detections[0]['box']
        rect = Rectangle((x, y), w, h, linewidth=3, edgecolor='lime', facecolor='none')
        ax24.add_patch(rect)
        ax24.text(x, y-10, formatted_text, fontsize=14, color='lime', fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax24.set_title('6. Kết quả cuối cùng', fontsize=11, fontweight='bold')
    ax24.axis('off')
    
    # Summary box
    ax25 = fig.add_subplot(gs[7, 2:6])
    summary_text = f"""
+==============================================================+
|                    KET QUA NHAN DANG                         |
+==============================================================+
|  Anh dau vao: {Path(image_path).name:<42} |
|  Bien so nhan dang: {formatted_text:<36} |
|  Do tin cay: {ocr_confidence:>6.1f}%                                   |
|  Format hop le: {'Co' if is_valid else 'Khong':<40} |
|  So ky tu: {len(ocr_text):<45} |
|  Loai bien: {plate_type:<44} |
+==============================================================+
"""
    ax25.text(0.05, 0.5, summary_text, transform=ax25.transAxes, fontsize=11,
              verticalalignment='center', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax25.axis('off')
    ax25.set_title('6. Danh gia ket qua', fontsize=11, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if output_path is None:
        output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_detailed_pipeline.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Đã lưu visualization: {output_path}")
    
    plt.close()
    
    return {
        'text': corrected_text,
        'confidence': ocr_confidence,
        'is_valid': is_valid,
        'formatted': formatted_text,
        'output_path': output_path
    }


def main():
    parser = argparse.ArgumentParser(description='Detailed Pipeline Visualization')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', help='Path to save visualization')
    parser.add_argument('--engine', choices=['tesseract', 'easyocr'], default='easyocr',
                        help='OCR engine to use')
    
    args = parser.parse_args()
    
    result = create_detailed_visualization(args.image, args.output, args.engine)
    
    if result:
        print("\n" + "=" * 70)
        print("HOÀN TẤT!")
        print("=" * 70)
        print(f"  🔤 Biển số: {result['formatted']}")
        print(f"  📊 Confidence: {result['confidence']:.1f}%")
        print(f"  📁 Output: {result['output_path']}")


if __name__ == '__main__':
    main()
