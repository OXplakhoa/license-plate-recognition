#!/usr/bin/env python3
"""
Ứng dụng Demo Streamlit - Nhận dạng Biển số Xe Việt Nam
==================================================================

Ứng dụng demo chuyên nghiệp để trình bày quy trình nhận dạng
biển số xe cho giảng viên và người đánh giá.

Cách chạy:
    streamlit run app.py

Yêu cầu:
    pip install streamlit opencv-python-headless pillow
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from src.pipeline import LicensePlateRecognizer
from src.lp_detector import (
    detect_with_character_validation,
    detect_multi_preset,
    correct_plate_perspective_and_skew,
)
from src.ocr_engine import configure_tesseract
from src.heuristics import apply_heuristics, is_valid_plate

# ============================================================================
# CACHED RESOURCES - Prevent reloading on every interaction
# ============================================================================

@st.cache_resource
def load_easyocr_reader():
    """Tải EasyOCR reader với caching. GPU tắt để tương thích macOS."""
    import easyocr
    # QUAN TRỌNG: gpu=False để tương thích macOS (không có CUDA)
    return easyocr.Reader(['en'], gpu=False)


@st.cache_resource
def get_recognizer(ocr_engine: str, hard_mode: bool, _version: int = 2):
    """Lấy instance recognizer đã cache. _version để force refresh cache."""
    return LicensePlateRecognizer(
        ocr_engine=ocr_engine,
        use_character_validation=True,
        use_perspective_correction=True,
        use_deskew=True,
        use_color_detection=hard_mode,
        use_mser_detection=hard_mode,
        debug=True
    )


# ============================================================================
# PREPROCESSING VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_preprocessing_steps(image: np.ndarray):
    """
    Trực quan hóa các bước tiền xử lý.
    
    Hàm này minh họa các bước xử lý ảnh được thực hiện
    trước khi phát hiện biển số xe.
    
    Args:
        image: Ảnh đầu vào định dạng BGR
        
    Returns:
        Dictionary chứa các ảnh đã xử lý cho từng bước
    """
    steps = {}
    
    # Bước 1: Chuyển sang Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps['grayscale'] = gray
    
    # Bước 2: Lọc nhiễu - Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    steps['blurred'] = blurred
    
    # Bước 3: Cân bằng độ tương phản - CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    steps['clahe'] = enhanced
    
    # Bước 4: Canny Edge Detection
    edges = cv2.Canny(enhanced, 50, 150)
    steps['edges'] = edges
    
    # Bước 5: Morphological operations (đóng, mở)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open)
    steps['morphology'] = morph
    
    # Bước 6: Tìm và vẽ contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    steps['contours'] = contour_img
    steps['contour_count'] = len(contours)
    
    # Lọc contours theo tỉ lệ biển số (aspect ratio ~ 1:1 đến 5:1)
    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img
    valid_contours = []
    plate_candidates = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Lọc theo diện tích và tỉ lệ
        if 0.005 < area / img_area < 0.3 and 0.5 < aspect_ratio < 6.0:
            valid_contours.append(cnt)
            plate_candidates.append((x, y, w, h, aspect_ratio))
    
    # Vẽ các candidate hợp lệ
    candidate_img = image.copy()
    for cnt in valid_contours:
        cv2.drawContours(candidate_img, [cnt], -1, (0, 255, 0), 2)
    for (x, y, w, h, ar) in plate_candidates:
        cv2.rectangle(candidate_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(candidate_img, f"AR:{ar:.1f}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    steps['candidates'] = candidate_img
    steps['candidate_count'] = len(plate_candidates)
    steps['plate_candidates'] = plate_candidates
    
    # Bước 8: Character Validation - Đánh giá từng candidate
    # Đây là bước quan trọng để loại bỏ false positives
    from src.lp_detector import compute_character_score
    
    scored_candidates = []
    for i, (x, y, w, h, ar) in enumerate(plate_candidates):
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            score, details = compute_character_score(roi)
            char_count = details.get('char_count', 0)
            scored_candidates.append({
                'index': i + 1,
                'box': (x, y, w, h),
                'aspect_ratio': ar,
                'char_score': score,
                'char_count': char_count,
                'is_valid': score >= 0.35  # Ngưỡng mặc định
            })
    
    # Sắp xếp theo điểm cao nhất
    scored_candidates.sort(key=lambda c: c['char_score'], reverse=True)
    
    # Vẽ ảnh với điểm character score
    scored_img = image.copy()
    for cand in scored_candidates:
        x, y, w, h = cand['box']
        score = cand['char_score']
        is_valid = cand['is_valid']
        
        # Màu xanh nếu valid, đỏ nếu không
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        cv2.rectangle(scored_img, (x, y), (x+w, y+h), color, 2)
        
        # Hiển thị score
        label = f"S:{score:.2f}"
        cv2.putText(scored_img, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    steps['scored_candidates'] = scored_img
    steps['scored_list'] = scored_candidates
    
    return steps


def visualize_plate_processing(plate_roi: np.ndarray):
    """
    Trực quan hóa các bước xử lý vùng biển số.
    
    Args:
        plate_roi: Ảnh vùng biển số đã cắt (BGR)
        
    Returns:
        Dictionary chứa các ảnh đã xử lý
    """
    steps = {}
    
    # Chuyển sang grayscale
    if len(plate_roi.shape) == 3:
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_roi
    steps['gray'] = gray
    
    # Resize nếu quá nhỏ
    h, w = gray.shape[:2]
    if h < 50:
        scale = 50 / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    steps['resized'] = gray
    
    # CLAHE - Cân bằng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    steps['clahe'] = enhanced
    
    # Otsu Threshold - Nhị phân hóa
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps['otsu'] = binary
    
    # Làm sạch nhiễu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps['cleaned'] = cleaned
    
    # Tìm contours ký tự
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contours theo kích thước ký tự
    char_h, char_w = cleaned.shape[:2]
    char_contours = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Ký tự thường có chiều cao từ 30-90% chiều cao biển số
        if 0.2 * char_h < h < 0.95 * char_h and w > 3:
            char_contours.append((x, y, w, h))
    
    # Sắp xếp từ trái sang phải
    char_contours.sort(key=lambda c: c[0])
    
    # Vẽ các ký tự đã segment
    if len(plate_roi.shape) == 3:
        segmented_img = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    else:
        segmented_img = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    for i, (x, y, w, h) in enumerate(char_contours):
        color = (0, 255, 0) if i % 2 == 0 else (0, 200, 255)
        cv2.rectangle(segmented_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(segmented_img, str(i+1), (x, y-3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    steps['segmented'] = segmented_img
    steps['char_count'] = len(char_contours)
    steps['char_boxes'] = char_contours
    
    # Trích xuất và resize từng ký tự thành 28x28
    char_images_28x28 = []
    for x, y, w, h in char_contours:
        # Cắt ký tự từ ảnh đã làm sạch
        char_img = cleaned[y:y+h, x:x+w]
        
        # Padding để giữ tỉ lệ trước khi resize
        # Tìm kích thước lớn hơn
        max_dim = max(h, w)
        
        # Tạo ảnh vuông với nền trắng (255)
        square_img = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
        
        # Đặt ký tự vào giữa
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        square_img[y_offset:y_offset+h, x_offset:x_offset+w] = char_img
        
        # Resize về 28x28
        char_28x28 = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
        char_images_28x28.append(char_28x28)
    
    steps['char_images_28x28'] = char_images_28x28
    
    return steps


def display_preprocessing_steps(steps: dict):
    """Hiển thị các bước tiền xử lý theo cột."""
    
    st.subheader("🔬 Trực quan hóa Pipeline - Các bước Tiền xử lý")
    
    # Hàng 1: Grayscale và Lọc nhiễu
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bước 1: Grayscale Conversion**")
        st.image(steps['grayscale'], caption="Ảnh Grayscale", use_container_width=True)
        st.caption("Chuyển ảnh màu BGR → Grayscale (thang xám)")
        
    with col2:
        st.markdown("**Bước 2: Gaussian Blur (Lọc nhiễu)**")
        st.image(steps['blurred'], caption="Ảnh sau Blur", use_container_width=True)
        st.caption("Kernel Gaussian 5x5 để giảm nhiễu")
    
    # Hàng 2: CLAHE và Canny
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Bước 3: CLAHE (Cân bằng độ tương phản)**")
        st.image(steps['clahe'], caption="Sau CLAHE", use_container_width=True)
        st.caption("Contrast Limited Adaptive Histogram Equalization")
        
    with col4:
        st.markdown("**Bước 4: Canny Edge Detection**")
        st.image(steps['edges'], caption="Bản đồ cạnh", use_container_width=True)
        st.caption("Phát hiện biên với ngưỡng Canny: 50-150")
    
    # Hàng 3: Morphology và Contours
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Bước 5: Morphological Operations**")
        st.image(steps['morphology'], caption="Sau Morphology", use_container_width=True)
        st.caption("Phép đóng (Close) + Phép mở (Open) để nối và làm sạch")
        
    with col6:
        st.markdown("**Bước 6: Contour Detection**")
        contour_rgb = cv2.cvtColor(steps['contours'], cv2.COLOR_BGR2RGB)
        st.image(contour_rgb, caption=f"Tổng số Contours: {steps['contour_count']}", use_container_width=True)
        st.caption("Tìm tất cả các đường viền trong ảnh")
    
    # Hàng 4: Lọc candidates
    st.markdown("**Bước 7: Lọc Candidates theo tỉ lệ khung**")
    candidate_rgb = cv2.cvtColor(steps['candidates'], cv2.COLOR_BGR2RGB)
    st.image(candidate_rgb, caption=f"Số vùng biển số tiềm năng: {steps['candidate_count']}", use_container_width=True)
    st.caption("Lọc contours có tỉ lệ W:H phù hợp (0.5 - 6.0) và diện tích hợp lệ")
    
    # Hiển thị thông tin candidates
    if steps['candidate_count'] > 0:
        with st.expander("📊 Chi tiết các vùng candidate"):
            for i, (x, y, w, h, ar) in enumerate(steps['plate_candidates'][:5]):
                st.write(f"**Candidate {i+1}:** Vị trí ({x}, {y}), Kích thước {w}x{h}, Aspect Ratio: {ar:.2f}")
    
    # Bước 8: Character Validation - QUAN TRỌNG
    st.markdown("**Bước 8: Character Validation (Xác thực ký tự) ⭐**")
    st.caption("🔑 **Bước quan trọng nhất!** Đánh giá xem mỗi candidate có chứa ký tự giống biển số không")
    
    if 'scored_candidates' in steps:
        scored_rgb = cv2.cvtColor(steps['scored_candidates'], cv2.COLOR_BGR2RGB)
        
        # Đếm valid candidates
        valid_count = sum(1 for c in steps['scored_list'] if c['is_valid'])
        st.image(scored_rgb, caption=f"🟢 Hợp lệ: {valid_count} | 🔴 Không hợp lệ: {len(steps['scored_list']) - valid_count}", use_container_width=True)
        
        # Giải thích
        st.info("""
        **Cách hoạt động:**
        - Mỗi candidate được phân tích để tìm các vùng giống ký tự (contours có tỉ lệ phù hợp)
        - **Character Score** được tính dựa trên: số lượng ký tự, độ đều, vị trí, kích thước
        - Ngưỡng mặc định: **Score ≥ 0.35** → Hợp lệ (màu xanh)
        - Candidate có score cao nhất sẽ được chọn làm biển số
        """)
        
        # Bảng chi tiết
        with st.expander("📊 Chi tiết điểm từng candidate", expanded=True):
            # Header
            cols = st.columns([1, 2, 2, 2, 2])
            cols[0].markdown("**#**")
            cols[1].markdown("**Kích thước**")
            cols[2].markdown("**Aspect Ratio**")
            cols[3].markdown("**Char Score**")
            cols[4].markdown("**Kết quả**")
            
            st.divider()
            
            # Rows - hiển thị top 5
            for cand in steps['scored_list'][:5]:
                cols = st.columns([1, 2, 2, 2, 2])
                x, y, w, h = cand['box']
                cols[0].write(f"{cand['index']}")
                cols[1].write(f"{w}×{h}")
                cols[2].write(f"{cand['aspect_ratio']:.2f}")
                
                # Score với màu
                score = cand['char_score']
                if cand['is_valid']:
                    cols[3].markdown(f"**:green[{score:.3f}]**")
                    cols[4].markdown("✅ **Hợp lệ**")
                else:
                    cols[3].markdown(f":red[{score:.3f}]")
                    cols[4].markdown("❌ Loại bỏ")


def display_plate_processing_steps(plate_steps: dict, plate_index: int = 1):
    """Hiển thị các bước xử lý vùng biển số."""
    
    st.markdown(f"#### 🔧 Chi tiết xử lý Biển số {plate_index}")
    
    with st.expander("Xem các bước Chuẩn hóa & Phân tách ký tự", expanded=False):
        # Hàng 1: Gray và CLAHE
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Grayscale + Resize**")
            st.image(plate_steps['resized'], caption="Ảnh chuẩn hóa kích thước", use_container_width=True)
            
        with col2:
            st.markdown("**CLAHE Enhancement**")
            st.image(plate_steps['clahe'], caption="Tăng độ tương phản", use_container_width=True)
        
        # Hàng 2: Otsu và Cleaned
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Otsu Threshold (Nhị phân hóa)**")
            st.image(plate_steps['otsu'], caption="Ảnh nhị phân", use_container_width=True)
            
        with col4:
            st.markdown("**Làm sạch nhiễu**")
            st.image(plate_steps['cleaned'], caption="Sau Morphology Open", use_container_width=True)
        
        # Segmentation result
        st.markdown("**Character Segmentation (Phân tách ký tự)**")
        st.image(plate_steps['segmented'], caption=f"Số ký tự phát hiện: {plate_steps['char_count']}", use_container_width=True)
        st.caption("Các hộp màu đánh dấu từng ký tự được tách ra")
        
        # Hiển thị ảnh 28x28 của từng ký tự
        char_images = plate_steps.get('char_images_28x28', [])
        if char_images:
            st.markdown("**📦 Ký tự 28×28 (chuẩn hóa)**")
            st.caption("Mỗi ký tự được resize về 28×28 pixel với padding giữ tỉ lệ - sẵn sàng cho OCR/ML")
            
            # Hiển thị tối đa 10 ký tự trên mỗi hàng
            num_chars = len(char_images)
            cols_per_row = min(10, num_chars)
            
            # Tạo các cột để hiển thị ký tự
            cols = st.columns(cols_per_row)
            for i, char_img in enumerate(char_images[:10]):
                with cols[i % cols_per_row]:
                    # Phóng to 4x để dễ nhìn (28 -> 112)
                    display_img = cv2.resize(char_img, (112, 112), interpolation=cv2.INTER_NEAREST)
                    st.image(display_img, caption=f"#{i+1}", use_container_width=True)
            
            # Nếu có nhiều hơn 10 ký tự, hiển thị hàng thứ 2
            if num_chars > 10:
                cols2 = st.columns(min(10, num_chars - 10))
                for i, char_img in enumerate(char_images[10:20]):
                    with cols2[i]:
                        display_img = cv2.resize(char_img, (112, 112), interpolation=cv2.INTER_NEAREST)
                        st.image(display_img, caption=f"#{i+11}", use_container_width=True)


# ============================================================================
# OCR FUNCTIONS
# ============================================================================

def ocr_with_tesseract(plate_image: np.ndarray) -> tuple:
    """
    Thực hiện OCR sử dụng Tesseract.
    
    Args:
        plate_image: Ảnh biển số đã cắt (grayscale hoặc BGR)
        
    Returns:
        (text, confidence)
    """
    import pytesseract
    
    configure_tesseract()
    
    # Đảm bảo ảnh grayscale
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image
    
    # Resize để OCR tốt hơn
    h, w = gray.shape[:2]
    if h < 50:
        scale = 50 / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Áp dụng CLAHE để tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Nhị phân hóa
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Thêm padding
    padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    
    # OCR với PSM 7 (một dòng văn bản)
    config = '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    try:
        data = pytesseract.image_to_data(padded, config=config, output_type=pytesseract.Output.DICT)
        
        texts = []
        confs = []
        for i, conf in enumerate(data['conf']):
            if int(conf) > 0:
                texts.append(data['text'][i])
                confs.append(int(conf))
        
        text = ''.join(texts).strip().upper()
        text = ''.join(c for c in text if c.isalnum())
        confidence = np.mean(confs) if confs else 0.0
        
        return text, confidence
    except Exception as e:
        st.error(f"Tesseract error: {e}")
        return "", 0.0


def ocr_with_easyocr(plate_image: np.ndarray) -> tuple:
    """
    Thực hiện OCR sử dụng EasyOCR.
    
    Args:
        plate_image: Ảnh biển số đã cắt
        
    Returns:
        (text, confidence)
    """
    reader = load_easyocr_reader()
    
    # Đảm bảo định dạng BGR
    if len(plate_image.shape) == 2:
        plate_bgr = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
    else:
        plate_bgr = plate_image
    
    # Resize để OCR tốt hơn
    h, w = plate_bgr.shape[:2]
    if h < 50:
        scale = 2.0
        plate_bgr = cv2.resize(plate_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Chuyển sang RGB cho EasyOCR
    plate_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
    
    # Danh sách ký tự cho phép
    allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    try:
        results = reader.readtext(plate_rgb, allowlist=allowlist, detail=1)
        
        if not results:
            return "", 0.0
        
        # Sắp xếp theo vị trí (trái sang phải, trên xuống dưới)
        results.sort(key=lambda x: (x[0][0][1] // 20, x[0][0][0]))
        
        texts = []
        confs = []
        for (bbox, text, conf) in results:
            texts.append(text)
            confs.append(conf)
        
        full_text = ''.join(texts).upper()
        full_text = ''.join(c for c in full_text if c.isalnum())
        avg_conf = np.mean(confs) * 100 if confs else 0.0
        
        return full_text, avg_conf
    except Exception as e:
        st.error(f"EasyOCR error: {e}")
        return "", 0.0


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Cấu hình trang
    st.set_page_config(
        page_title="Nhận dạng Biển số Xe Việt Nam",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Tiêu đề
    st.title("🚗 Nhận dạng Biển số Xe Việt Nam")
    st.markdown("""
    **Pipeline Xử lý Ảnh Truyền thống** sử dụng OpenCV + Tesseract/EasyOCR  
    *Không dùng Deep Learning cho Phát hiện - Xử lý Ảnh Thuần túy*
    """)
    
    st.divider()
    
    # ========================================================================
    # SIDEBAR CONFIGURATION
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Chọn nguồn đầu vào
        st.subheader("📷 Nguồn đầu vào")
        input_source = st.radio(
            "Chọn phương thức nhập:",
            ["Tải ảnh lên (JPG/PNG)", "Dán từ Clipboard", "Camera trực tiếp"],
            index=0
        )
        
        st.divider()
        
        # Chọn OCR Engine
        st.subheader("🔤 OCR Engine")
        ocr_engine = st.selectbox(
            "Chọn engine OCR:",
            ["tesseract", "easyocr"],
            index=1,  # Mặc định EasyOCR
            format_func=lambda x: "Tesseract (Nhanh)" if x == "tesseract" else "EasyOCR (Chính xác)"
        )
        
        st.divider()
        
        # Tùy chọn nâng cao
        st.subheader("🔧 Tùy chọn nâng cao")
        hard_mode = st.checkbox(
            "Chế độ Khó",
            value=False,
            help="Bật thêm các phương pháp phát hiện (Color + MSER) cho ảnh khó"
        )
        
        show_preprocessing = st.checkbox(
            "Hiển thị các bước Tiền xử lý",
            value=True,
            help="Hiển thị các bước xử lý trung gian"
        )
        
        st.divider()
        
        # Thông tin
        st.info("""
        **Các bước Pipeline:**
        
        📥 **1. Nhập ảnh đầu vào**
        - JPG, PNG hoặc Camera
        
        🔧 **2. Tiền xử lý (Preprocessing)**
        - Grayscale conversion
        - Gaussian Blur (lọc nhiễu)
        - CLAHE (cân bằng tương phản)
        
        🔍 **3. Phát hiện vùng biển số**
        - Canny Edge Detection
        - Morphology (đóng, mở)
        - Contour Detection
        - Lọc theo tỉ lệ khung
        
        ✂️ **4. Chuẩn hóa & Phân tách**
        - Otsu Threshold (nhị phân)
        - Character Segmentation
        
        🔤 **5. Nhận dạng OCR**
        - Tesseract hoặc EasyOCR
        """)
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    image = None
    
    # Xử lý các nguồn đầu vào khác nhau
    if input_source == "Tải ảnh lên (JPG/PNG)":
        uploaded_file = st.file_uploader(
            "Tải lên ảnh chứa xe có biển số",
            type=['jpg', 'jpeg', 'png'],
            help="Định dạng hỗ trợ: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Đọc ảnh đã tải lên
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    elif input_source == "Dán từ Clipboard":
        st.info("📋 Dán ảnh Base64 hoặc URL vào ô bên dưới")
        
        # Hướng dẫn cách lấy Base64
        with st.expander("📖 Hướng dẫn dán ảnh", expanded=False):
            st.markdown("""
            **Cách 1: Dán URL ảnh trực tiếp**
            - Copy URL ảnh (ví dụ: `https://example.com/image.jpg`)
            - Dán vào ô bên dưới
            
            **Cách 2: Dán ảnh Base64**
            1. Copy ảnh vào clipboard (Ctrl+C hoặc Screenshot)
            2. Mở trang web: [Base64 Image Encoder](https://www.base64-image.de/)
            3. Dán ảnh (Ctrl+V) và copy chuỗi Base64
            4. Dán vào ô bên dưới
            
            **Cách 3: Kéo thả file** (Khuyên dùng)
            - Chuyển sang tab "Tải ảnh lên" và kéo thả file vào
            """)
        
        # Text input để nhận base64/URL
        clipboard_input = st.text_area(
            "Dán Base64 hoặc URL ảnh:",
            height=120,
            placeholder="Dán vào đây:\n• data:image/png;base64,iVBORw0KGgo...\n• https://example.com/car.jpg",
            key="clipboard_input"
        )
        
        if clipboard_input and clipboard_input.strip():
            clipboard_input = clipboard_input.strip()
            try:
                if clipboard_input.startswith('data:image'):
                    # Base64 image
                    import base64
                    # Tách phần base64 data
                    if ',' in clipboard_input:
                        base64_data = clipboard_input.split(',')[1]
                    else:
                        base64_data = clipboard_input
                    img_bytes = base64.b64decode(base64_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                elif clipboard_input.startswith('http'):
                    # URL image
                    import urllib.request
                    with st.spinner("Đang tải ảnh từ URL..."):
                        req = urllib.request.Request(
                            clipboard_input,
                            headers={'User-Agent': 'Mozilla/5.0'}
                        )
                        with urllib.request.urlopen(req, timeout=10) as response:
                            img_bytes = response.read()
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    # Thử decode như base64 thuần
                    import base64
                    try:
                        img_bytes = base64.b64decode(clipboard_input)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    except:
                        st.error("❌ Định dạng không hợp lệ. Vui lòng dán URL hoặc Base64.")
                        image = None
                    
                if image is not None:
                    st.success("✅ Đã tải ảnh thành công!")
                else:
                    st.error("❌ Không thể đọc ảnh. Vui lòng kiểm tra lại dữ liệu.")
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý ảnh: {str(e)}")
            
    else:  # Camera trực tiếp
        st.warning("📹 Đầu vào Camera - Nhấn 'Chụp ảnh' để chụp")
        camera_input = st.camera_input("Chụp ảnh từ camera")
        
        if camera_input is not None:
            file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # ========================================================================
    # PROCESSING PIPELINE
    # ========================================================================
    
    if image is not None:
        # Hiển thị ảnh gốc
        st.subheader("📸 Ảnh gốc")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Ảnh đầu vào", use_container_width=True)
        
        st.divider()
        
        # Hiển thị các bước tiền xử lý nếu được bật
        if show_preprocessing:
            with st.spinner("Đang trực quan hóa các bước tiền xử lý..."):
                steps = visualize_preprocessing_steps(image)
                display_preprocessing_steps(steps)
            
            st.divider()
        
        # Chạy phát hiện
        st.subheader("🔍 Phát hiện & Nhận dạng")
        
        with st.spinner(f"Đang chạy phát hiện với {ocr_engine.upper()}..."):
            try:
                # Get recognizer
                recognizer = get_recognizer(ocr_engine, hard_mode)
                
                # Run full pipeline
                result = recognizer.recognize(image)
                
                if result.plates:
                    st.success(f"✅ Tìm thấy {len(result.plates)} biển số xe!")
                    
                    # Xử lý từng biển số phát hiện được
                    for i, plate in enumerate(result.plates):
                        st.markdown(f"### 🚙 Biển số {i + 1}")
                        
                        col_plate, col_result = st.columns([1, 2])
                        
                        with col_plate:
                            # Trích xuất và hiển thị vùng biển số
                            x, y, w, h = plate.box
                            plate_roi = image[y:y+h, x:x+w]
                            plate_roi_rgb = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB)
                            
                            st.image(plate_roi_rgb, caption="Vùng biển số đã cắt", use_container_width=True)
                            
                            # Hiển thị ảnh đã chỉnh nếu có
                            if plate.corrected_image is not None:
                                st.image(plate.corrected_image, caption="Sau Perspective Correction", use_container_width=True)
                        
                        with col_result:
                            # Hiển thị kết quả nhận dạng
                            st.markdown("#### Kết quả Nhận dạng")
                            
                            # Hiển thị text lớn
                            if plate.text:
                                st.markdown(f"""
                                <div style="
                                    background-color: #1e3a1e;
                                    padding: 20px;
                                    border-radius: 10px;
                                    text-align: center;
                                    margin: 10px 0;
                                ">
                                    <span style="
                                        font-size: 48px;
                                        font-weight: bold;
                                        font-family: monospace;
                                        color: #00ff00;
                                        letter-spacing: 5px;
                                    ">{plate.text}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning("Không thể nhận dạng văn bản")
                            
                            # Đo độ tin cậy
                            st.metric(
                                label="Độ tin cậy",
                                value=f"{plate.confidence:.1f}%",
                                delta="Cao" if plate.confidence > 80 else ("Trung bình" if plate.confidence > 50 else "Thấp")
                            )
                            
                            # Thông tin bổ sung
                            st.markdown(f"""
                            | Thuộc tính | Giá trị |
                            |----------|-------|
                            | **Loại biển số** | {plate.plate_type} |
                            | **Phương pháp phát hiện** | {plate.detection_method} |
                            | **Bounding Box** | {plate.box} |
                            """)
                        
                        # Hiển thị chi tiết xử lý biển số
                        plate_steps = visualize_plate_processing(plate_roi)
                        display_plate_processing_steps(plate_steps, i + 1)
                        
                        st.divider()
                    
                    # Vẽ tất cả các phát hiện lên ảnh
                    st.subheader("📍 Trực quan hóa Phát hiện")
                    result_img = image.copy()
                    for plate in result.plates:
                        x, y, w, h = plate.box
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        
                        # Thêm nhãn text
                        label = f"{plate.text} ({plate.confidence:.0f}%)"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(result_img, (x, y-th-10), (x+tw+10, y), (0, 255, 0), -1)
                        cv2.putText(result_img, label, (x+5, y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_img_rgb, caption="Kết quả Phát hiện", use_container_width=True)
                    
                    # Thời gian xử lý
                    st.info(f"⏱️ Thời gian xử lý: {result.processing_time_ms:.1f} ms")
                    
                else:
                    st.warning("⚠️ Không phát hiện biển số xe trong ảnh này.")
                    st.markdown("""
                    **Mẹo để phát hiện tốt hơn:**
                    - Đảm bảo biển số rõ ràng và không quá nhỏ
                    - Thử bật "Chế độ Khó" cho ảnh khó
                    - Đảm bảo ảnh có ánh sáng tốt
                    - Biển số không nên bị nghiêng nhiều hoặc che khuất
                    """)
                    
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình xử lý: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    else:
        # Placeholder khi chưa có ảnh
        st.info("👆 Vui lòng tải ảnh lên hoặc chụp từ camera để bắt đầu.")
        
        # Phần ảnh mẫu
        st.subheader("📁 Ảnh mẫu")
        st.markdown("Bạn có thể test với các ảnh từ thư mục `data/test_images/`:")
        
        sample_folder = Path("data/test_images/CarTGMT")
        if sample_folder.exists():
            sample_images = list(sample_folder.glob("*.jpg"))[:5]
            
            if sample_images:
                cols = st.columns(5)
                for i, img_path in enumerate(sample_images):
                    with cols[i]:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_small = cv2.resize(img_rgb, (150, 100))
                            st.image(img_small, caption=img_path.name[:15] + "...")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
        Hệ thống Nhận dạng Biển số Xe Việt Nam<br>
        Xử lý Ảnh Truyền thống (OpenCV + Tesseract/EasyOCR)<br>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()