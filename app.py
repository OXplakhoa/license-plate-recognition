#!/usr/bin/env python3
"""
Ứng dụng Demo Streamlit - Nhận dạng Biển số Xe Việt Nam
==================================================================

Cách chạy:
    streamlit run app.py

Yêu cầu:
    pip install streamlit opencv-python-headless pillow
"""

import traceback

import cv2
import streamlit as st
from pathlib import Path

from src.pipeline import LicensePlateRecognizer
from src.visualization import visualize_preprocessing_steps, visualize_plate_processing
from app_components import (
    display_preprocessing_steps,
    display_plate_processing_steps,
    display_plate_result,
    display_detection_overlay,
    display_no_detection_tips,
    display_sample_images,
)
from app_input import get_input_image


# ============================================================================
# CACHED RESOURCES
# ============================================================================

@st.cache_resource
def get_recognizer(ocr_engine: str, hard_mode: bool, _version: int = 2):
    """Return a cached ``LicensePlateRecognizer`` instance."""
    return LicensePlateRecognizer(
        ocr_engine=ocr_engine,
        use_character_validation=True,
        use_perspective_correction=True,
        use_deskew=True,
        use_color_detection=hard_mode,
        use_mser_detection=hard_mode,
        debug=True,
    )


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar() -> tuple:
    """Draw sidebar widgets and return ``(input_source, ocr_engine, hard_mode, show_preprocessing)``."""
    with st.sidebar:
        st.header("⚙️ Cấu hình")

        st.subheader("📷 Nguồn đầu vào")
        input_source = st.radio(
            "Chọn phương thức nhập:",
            ["Tải ảnh lên (JPG/PNG)", "Dán từ Clipboard", "Camera trực tiếp"],
            index=0,
        )
        st.divider()

        st.subheader("🔤 OCR Engine")
        ocr_engine = st.selectbox(
            "Chọn engine OCR:",
            ["tesseract", "easyocr"],
            index=1,
            format_func=lambda x: "Tesseract (Nhanh)" if x == "tesseract" else "EasyOCR (Chính xác)",
        )
        st.divider()

        st.subheader("🔧 Tùy chọn nâng cao")
        hard_mode = st.checkbox(
            "Chế độ Khó", value=False,
            help="Bật thêm các phương pháp phát hiện (Color + MSER) cho ảnh khó",
        )
        show_preprocessing = st.checkbox(
            "Hiển thị các bước Tiền xử lý", value=True,
            help="Hiển thị các bước xử lý trung gian",
        )
        st.divider()

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

    return input_source, ocr_engine, hard_mode, show_preprocessing


# ============================================================================
# PROCESSING
# ============================================================================

def process_image(image, ocr_engine: str, hard_mode: bool, show_preprocessing: bool):
    """Run detection pipeline and render all results."""
    # Original image
    st.subheader("📸 Ảnh gốc")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Ảnh đầu vào", use_container_width=True)
    st.divider()

    # Optional preprocessing visualisation
    if show_preprocessing:
        with st.spinner("Đang trực quan hóa các bước tiền xử lý..."):
            steps = visualize_preprocessing_steps(image)
            display_preprocessing_steps(steps)
        st.divider()

    # Detection + OCR
    st.subheader("🔍 Phát hiện & Nhận dạng")

    with st.spinner(f"Đang chạy phát hiện với {ocr_engine.upper()}..."):
        try:
            recognizer = get_recognizer(ocr_engine, hard_mode)
            result = recognizer.recognize(image)

            if result.plates:
                st.success(f"✅ Tìm thấy {len(result.plates)} biển số xe!")
                _render_plates(image, result)
            else:
                display_no_detection_tips()

        except Exception as exc:
            st.error(f"❌ Lỗi trong quá trình xử lý: {exc}")
            st.code(traceback.format_exc())


def _render_plates(image, result):
    """Render per-plate details + final overlay."""
    for i, plate in enumerate(result.plates):
        st.markdown(f"### 🚙 Biển số {i + 1}")
        x, y, w, h = plate.box
        plate_roi = image[y : y + h, x : x + w]
        plate_roi_rgb = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB)

        display_plate_result(plate, plate_roi_rgb)

        plate_steps = visualize_plate_processing(plate_roi)
        display_plate_processing_steps(plate_steps, i + 1)
        st.divider()

    display_detection_overlay(image, result.plates)
    st.info(f"⏱️ Thời gian xử lý: {result.processing_time_ms:.1f} ms")


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="Nhận dạng Biển số Xe Việt Nam",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚗 Nhận dạng Biển số Xe Việt Nam")
    st.markdown(
        "**Pipeline Xử lý Ảnh Truyền thống** sử dụng OpenCV + Tesseract/EasyOCR  \n"
        "*Không dùng Deep Learning cho Phát hiện - Xử lý Ảnh Thuần túy*"
    )
    st.divider()

    input_source, ocr_engine, hard_mode, show_preprocessing = render_sidebar()

    image = get_input_image(input_source)

    if image is not None:
        process_image(image, ocr_engine, hard_mode, show_preprocessing)
    else:
        st.info("👆 Vui lòng tải ảnh lên hoặc chụp từ camera để bắt đầu.")
        st.subheader("📁 Ảnh mẫu")
        st.markdown("Bạn có thể test với các ảnh mẫu bên dưới hoặc tải ảnh của bạn lên:")
        sample_folder = Path("data/samples")
        if not sample_folder.exists():
            sample_folder = Path("data/test_images/CarTGMT")
        display_sample_images(sample_folder)

    # Footer
    st.divider()
    st.markdown(
        '<div style="text-align: center; color: gray; font-size: 12px;">'
        "Hệ thống Nhận dạng Biển số Xe Việt Nam<br>"
        "Xử lý Ảnh Truyền thống (OpenCV + Tesseract/EasyOCR)<br>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()