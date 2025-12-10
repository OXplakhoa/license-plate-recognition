#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Pipeline Steps - Hiển thị chi tiết từng bước xử lý nhận diện biển số
================================================================================

Script này sẽ hiển thị:
1. Ảnh gốc
2. Chuyển đổi grayscale  
3. Phát hiện vùng biển số (detection)
4. Trích xuất ROI (Region of Interest)
5. Tiền xử lý ROI (preprocessing)
6. Kết quả OCR

Cách sử dụng:
    python scripts/visualize_pipeline_steps.py --image path/to/image.jpg
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.utils import ensure_grayscale
from src.lp_detector import (
    detect_with_character_validation,
    detect_multi_preset,
    correct_plate_perspective_and_skew,
    get_plate_type_from_aspect,
    detect_character_candidates,
    compute_character_score,
)
from src.character_segmenter import segment_characters, binarize_plate


def create_visualization(image_path: str, output_path: str = "pipeline_steps.png", use_easyocr: bool = True):
    """
    Tạo visualization chi tiết từng bước của pipeline.
    """
    # ========================================
    # BƯỚC 1: Đọc ảnh gốc
    # ========================================
    print("\n" + "="*60)
    print("BƯỚC 1: Đọc ảnh gốc")
    print("="*60)
    
    original = cv2.imread(image_path)
    if original is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return
    
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    print(f"✓ Kích thước ảnh: {original.shape[1]}x{original.shape[0]} pixels")
    print(f"✓ Số kênh màu: {original.shape[2]} (BGR)")
    
    # ========================================
    # BƯỚC 2: Chuyển đổi Grayscale
    # ========================================
    print("\n" + "="*60)
    print("BƯỚC 2: Chuyển đổi sang ảnh xám (Grayscale)")
    print("="*60)
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    print(f"✓ Chuyển từ 3 kênh (BGR) → 1 kênh (Gray)")
    print(f"✓ Giảm kích thước dữ liệu: {original.shape[1]*original.shape[0]*3} → {gray.shape[1]*gray.shape[0]} bytes")
    
    # ========================================
    # BƯỚC 3: Phát hiện vùng biển số
    # ========================================
    print("\n" + "="*60)
    print("BƯỚC 3: Phát hiện vùng biển số (Detection)")
    print("="*60)
    
    # Sử dụng multi-preset detection để tìm các vùng ứng viên
    detections, detect_info = detect_with_character_validation(
        gray, 
        min_char_score=0.35,
        debug=True
    )
    
    print(f"✓ Phương pháp: Multi-preset contour detection + Character validation")
    print(f"✓ Số ứng viên tìm được: {len(detections)}")
    
    # Tạo ảnh hiển thị các vùng detection
    detection_vis = original_rgb.copy()
    
    if detections:
        best_det = detections[0]
        x, y, w, h = best_det['box']
        
        # Vẽ tất cả detections (màu vàng)
        for i, det in enumerate(detections):
            dx, dy, dw, dh = det['box']
            color = (0, 255, 0) if i == 0 else (255, 255, 0)  # Xanh cho best, vàng cho others
            cv2.rectangle(detection_vis, (dx, dy), (dx+dw, dy+dh), color, 2)
            cv2.putText(detection_vis, f"#{i+1}", (dx, dy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"\n📍 Vùng biển số tốt nhất:")
        print(f"   - Vị trí: ({x}, {y})")
        print(f"   - Kích thước: {w}x{h} pixels")
        print(f"   - Điểm char_score: {best_det.get('char_score', 0):.2f}")
        print(f"   - Số ký tự phát hiện: {best_det.get('char_count', 0)}")
    else:
        print("❌ Không tìm thấy vùng biển số nào!")
        return
    
    # ========================================
    # BƯỚC 4: Trích xuất ROI
    # ========================================
    print("\n" + "="*60)
    print("BƯỚC 4: Trích xuất vùng biển số (ROI)")
    print("="*60)
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_bgr = original[y:y+h, x:x+w]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    
    aspect_ratio = w / float(h) if h > 0 else 1.0
    plate_type = get_plate_type_from_aspect(aspect_ratio)
    
    print(f"✓ Trích xuất ROI tại vị trí ({x}, {y}) với kích thước {w}x{h}")
    print(f"✓ Tỷ lệ khung hình (Aspect Ratio): {aspect_ratio:.2f}")
    print(f"✓ Loại biển số: {plate_type}")
    if plate_type == "car_square" or plate_type == "car2":
        print("   → Biển ô tô vuông 2 dòng")
    elif plate_type == "car_rect" or plate_type == "car1":
        print("   → Biển ô tô chữ nhật 1 dòng")
    elif plate_type == "bike":
        print("   → Biển xe máy")
    
    # ========================================
    # BƯỚC 5: Tiền xử lý ROI
    # ========================================
    print("\n" + "="*60)
    print("BƯỚC 5: Tiền xử lý ROI (Preprocessing)")
    print("="*60)
    
    # 5.1 CLAHE - Tăng cường độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(roi_gray)
    print("✓ Bước 5.1: CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    print("   → Tăng cường độ tương phản cục bộ")
    
    # 5.2 Gaussian Blur - Giảm nhiễu
    blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0)
    print("✓ Bước 5.2: Gaussian Blur (3x3)")
    print("   → Giảm nhiễu, làm mịn ảnh")
    
    # 5.3 Binarization - Nhị phân hóa (với aggressive=False để giữ detail)
    binary, inverted = binarize_plate(roi_gray, method="combined", denoise=True)
    print("✓ Bước 5.3: Binarization (Otsu + Adaptive, light morphology)")
    print("   → Chuyển đổi sang ảnh nhị phân, giữ lại chi tiết ký tự")
    
    # 5.4 Perspective correction
    roi_box = (0, 0, roi_gray.shape[1], roi_gray.shape[0])
    corrected, _ = correct_plate_perspective_and_skew(roi_gray, roi=roi_box, deskew=True)
    if corrected is None:
        corrected = roi_gray
    print("✓ Bước 5.4: Perspective Correction + Deskew")
    print("   → Sửa góc nghiêng và hiệu chỉnh phối cảnh")
    
    # ========================================
    # BƯỚC 6: Phân đoạn ký tự (Character Segmentation)
    # ========================================
    print("\n" + "="*60)
    print("BƯỚC 6: Phân đoạn ký tự (Character Segmentation)")
    print("="*60)
    
    seg_result = segment_characters(corrected, plate_type=plate_type, debug=True)
    
    print(f"✓ Số ký tự phát hiện: {len(seg_result.boxes)}")
    print(f"✓ Phương pháp binarization: {seg_result.debug_info.get('method', 'combined')}")
    
    # Tạo visualization cho character segmentation
    char_vis = cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB)
    for i, box in enumerate(seg_result.boxes):
        bx, by, bw, bh = box
        cv2.rectangle(char_vis, (int(bx), int(by)), (int(bx+bw), int(by+bh)), (0, 255, 0), 1)
        cv2.putText(char_vis, str(i+1), (int(bx)+2, int(by)+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    print("\n📦 Chi tiết các ký tự:")
    for i, box in enumerate(seg_result.boxes):
        bx, by, bw, bh = box
        print(f"   Ký tự {i+1}: vị trí=({bx},{by}), kích thước={bw}x{bh}")
    
    # ========================================
    # BƯỚC 7: OCR - Nhận dạng ký tự
    # ========================================
    print("\n" + "="*60)
    print("BƯỚC 7: OCR - Nhận dạng ký tự quang học")
    print("="*60)
    
    if use_easyocr:
        print("✓ Sử dụng EasyOCR engine")
        print("   → Deep Learning based OCR")
        print("   → Hỗ trợ nhận dạng tiếng Việt")
        
        try:
            from src.ocr_easy import ocr_plate_easyocr
            text, confidence, char_confs = ocr_plate_easyocr(corrected)
            print(f"\n🔤 Kết quả OCR:")
            print(f"   - Text: {text}")
            print(f"   - Confidence: {confidence:.2f}%")
        except Exception as e:
            print(f"❌ Lỗi OCR: {e}")
            text = ""
            confidence = 0
    else:
        print("✓ Sử dụng Tesseract OCR engine")
        from src.ocr_engine import ocr_characters
        
        ocr_result = ocr_characters(seg_result.char_images, vn_plate=True, plate_type=plate_type)
        text = ocr_result.text
        confidence = ocr_result.mean_conf
        
        print(f"\n🔤 Kết quả OCR:")
        print(f"   - Text: {text}")
        print(f"   - Mean Confidence: {confidence:.2f}%")
        print(f"   - Per-char confidences: {[f'{c:.0f}%' for c in ocr_result.confidences]}")
    
    # ========================================
    # BƯỚC 8: Hậu xử lý (Post-processing)
    # ========================================
    print("\n" + "="*60)
    print("BƯỚC 8: Hậu xử lý (Post-processing)")
    print("="*60)
    
    from src.heuristics import apply_heuristics, is_valid_plate
    
    final_text = apply_heuristics(text)
    is_valid = is_valid_plate(final_text)
    
    print(f"✓ Text trước heuristics: {text}")
    print(f"✓ Text sau heuristics: {final_text}")
    print(f"✓ Biển số hợp lệ: {'✓ Có' if is_valid else '✗ Không'}")
    
    # ========================================
    # TẠO VISUALIZATION
    # ========================================
    print("\n" + "="*60)
    print("Tạo visualization...")
    print("="*60)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Pipeline nhận dạng biển số: {final_text} ({confidence:.1f}%)', fontsize=14, fontweight='bold')
    
    # Row 1
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('1. Ảnh gốc (Original)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('2. Grayscale')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(detection_vis)
    axes[0, 2].set_title('3. Detection (Phát hiện vùng biển số)')
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(roi_rgb)
    axes[1, 0].set_title(f'4. ROI ({w}x{h}px, AR={aspect_ratio:.2f})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(clahe_img, cmap='gray')
    axes[1, 1].set_title('5a. CLAHE (Tăng tương phản)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(binary, cmap='gray')
    axes[1, 2].set_title('5b. Binary (Nhị phân hóa)')
    axes[1, 2].axis('off')
    
    # Row 3
    axes[2, 0].imshow(corrected, cmap='gray')
    axes[2, 0].set_title('6. Corrected (Sửa góc nghiêng)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(char_vis)
    axes[2, 1].set_title(f'7. Segmentation ({len(seg_result.boxes)} ký tự)')
    axes[2, 1].axis('off')
    
    # Final result
    result_img = original_rgb.copy()
    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # Add text
    label = f"{final_text} ({confidence:.1f}%)"
    cv2.putText(result_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    axes[2, 2].imshow(result_img)
    axes[2, 2].set_title(f'8. Kết quả: {final_text}')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Đã lưu visualization: {output_path}")
    
    # Hiển thị
    plt.show()
    
    return {
        'text': final_text,
        'confidence': confidence,
        'box': (x, y, w, h),
        'plate_type': plate_type,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize License Plate Recognition Pipeline Steps")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="pipeline_steps.png", help="Path to save visualization")
    parser.add_argument("--engine", default="easyocr", choices=["tesseract", "easyocr"], 
                        help="OCR engine to use")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Không tìm thấy file: {args.image}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("VISUALIZATION PIPELINE NHẬN DIỆN BIỂN SỐ XE")
    print("="*60)
    print(f"📷 Ảnh đầu vào: {args.image}")
    print(f"🔧 OCR Engine: {args.engine}")
    
    result = create_visualization(
        args.image, 
        args.output,
        use_easyocr=(args.engine == "easyocr")
    )
    
    if result:
        print("\n" + "="*60)
        print("📋 KẾT QUẢ CUỐI CÙNG")
        print("="*60)
        print(f"   🔤 Biển số: {result['text']}")
        print(f"   📊 Độ tin cậy: {result['confidence']:.2f}%")
        print(f"   📍 Vị trí: {result['box']}")
        print(f"   🏷️ Loại biển: {result['plate_type']}")
        print("="*60)


if __name__ == "__main__":
    main()
