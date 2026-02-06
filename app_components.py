"""
Streamlit UI components for the license-plate recognition demo.

Every function here is a *pure display* helper — it receives pre-computed
data (dicts / images) and renders them with ``streamlit`` widgets.
No heavy image-processing logic belongs here.
"""

import cv2
import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# Preprocessing pipeline steps
# ---------------------------------------------------------------------------

def display_preprocessing_steps(steps: dict) -> None:
    """Render the full-image preprocessing steps dictionary."""

    st.subheader("🔬 Trực quan hóa Pipeline - Các bước Tiền xử lý")

    # Row 1 – Grayscale & Blur
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bước 1: Grayscale Conversion**")
        st.image(steps["grayscale"], caption="Ảnh Grayscale", use_container_width=True)
        st.caption("Chuyển ảnh màu BGR → Grayscale (thang xám)")
    with col2:
        st.markdown("**Bước 2: Gaussian Blur (Lọc nhiễu)**")
        st.image(steps["blurred"], caption="Ảnh sau Blur", use_container_width=True)
        st.caption("Kernel Gaussian 5x5 để giảm nhiễu")

    # Row 2 – CLAHE & Canny
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Bước 3: CLAHE (Cân bằng độ tương phản)**")
        st.image(steps["clahe"], caption="Sau CLAHE", use_container_width=True)
        st.caption("Contrast Limited Adaptive Histogram Equalization")
    with col4:
        st.markdown("**Bước 4: Canny Edge Detection**")
        st.image(steps["edges"], caption="Bản đồ cạnh", use_container_width=True)
        st.caption("Phát hiện biên với ngưỡng Canny: 50-150")

    # Row 3 – Morphology & Contours
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Bước 5: Morphological Operations**")
        st.image(steps["morphology"], caption="Sau Morphology", use_container_width=True)
        st.caption("Phép đóng (Close) + Phép mở (Open) để nối và làm sạch")
    with col6:
        st.markdown("**Bước 6: Contour Detection**")
        contour_rgb = cv2.cvtColor(steps["contours"], cv2.COLOR_BGR2RGB)
        st.image(contour_rgb,
                 caption=f"Tổng số Contours: {steps['contour_count']}",
                 use_container_width=True)
        st.caption("Tìm tất cả các đường viền trong ảnh")

    # Row 4 – Candidate filtering
    st.markdown("**Bước 7: Lọc Candidates theo tỉ lệ khung**")
    candidate_rgb = cv2.cvtColor(steps["candidates"], cv2.COLOR_BGR2RGB)
    st.image(candidate_rgb,
             caption=f"Số vùng biển số tiềm năng: {steps['candidate_count']}",
             use_container_width=True)
    st.caption("Lọc contours có tỉ lệ W:H phù hợp (0.5 - 6.0) và diện tích hợp lệ")

    if steps["candidate_count"] > 0:
        with st.expander("📊 Chi tiết các vùng candidate"):
            for i, (x, y, w, h, ar) in enumerate(steps["plate_candidates"][:5]):
                st.write(
                    f"**Candidate {i+1}:** Vị trí ({x}, {y}), "
                    f"Kích thước {w}x{h}, Aspect Ratio: {ar:.2f}"
                )

    # Row 5 – Character-score validation
    _display_character_validation(steps)


def _display_character_validation(steps: dict) -> None:
    """Render the character-score validation sub-section."""

    st.markdown("**Bước 8: Character Validation (Xác thực ký tự) ⭐**")
    st.caption(
        "🔑 **Bước quan trọng nhất!** Đánh giá xem mỗi candidate "
        "có chứa ký tự giống biển số không"
    )

    if "scored_candidates" not in steps:
        return

    scored_rgb = cv2.cvtColor(steps["scored_candidates"], cv2.COLOR_BGR2RGB)
    valid_count = sum(1 for c in steps["scored_list"] if c["is_valid"])
    invalid_count = len(steps["scored_list"]) - valid_count

    st.image(
        scored_rgb,
        caption=f"🟢 Hợp lệ: {valid_count} | 🔴 Không hợp lệ: {invalid_count}",
        use_container_width=True,
    )

    st.info("""
    **Cách hoạt động:**
    - Mỗi candidate được phân tích để tìm các vùng giống ký tự (contours có tỉ lệ phù hợp)
    - **Character Score** được tính dựa trên: số lượng ký tự, độ đều, vị trí, kích thước
    - Ngưỡng mặc định: **Score ≥ 0.35** → Hợp lệ (màu xanh)
    - Candidate có score cao nhất sẽ được chọn làm biển số
    """)

    with st.expander("📊 Chi tiết điểm từng candidate", expanded=True):
        cols = st.columns([1, 2, 2, 2, 2])
        cols[0].markdown("**#**")
        cols[1].markdown("**Kích thước**")
        cols[2].markdown("**Aspect Ratio**")
        cols[3].markdown("**Char Score**")
        cols[4].markdown("**Kết quả**")
        st.divider()

        for cand in steps["scored_list"][:5]:
            cols = st.columns([1, 2, 2, 2, 2])
            x, y, w, h = cand["box"]
            cols[0].write(f"{cand['index']}")
            cols[1].write(f"{w}×{h}")
            cols[2].write(f"{cand['aspect_ratio']:.2f}")

            score = cand["char_score"]
            if cand["is_valid"]:
                cols[3].markdown(f"**:green[{score:.3f}]**")
                cols[4].markdown("✅ **Hợp lệ**")
            else:
                cols[3].markdown(f":red[{score:.3f}]")
                cols[4].markdown("❌ Loại bỏ")


# ---------------------------------------------------------------------------
# Plate-ROI processing steps
# ---------------------------------------------------------------------------

def display_plate_processing_steps(plate_steps: dict, plate_index: int = 1) -> None:
    """Render segmentation / binarization details for a single plate."""

    st.markdown(f"#### 🔧 Chi tiết xử lý Biển số {plate_index}")

    with st.expander("Xem các bước Chuẩn hóa & Phân tách ký tự", expanded=False):
        # Gray + CLAHE
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Grayscale + Resize**")
            st.image(plate_steps["resized"], caption="Ảnh chuẩn hóa kích thước",
                     use_container_width=True)
        with col2:
            st.markdown("**CLAHE Enhancement**")
            st.image(plate_steps["clahe"], caption="Tăng độ tương phản",
                     use_container_width=True)

        # Otsu + Cleaned
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Otsu Threshold (Nhị phân hóa)**")
            st.image(plate_steps["otsu"], caption="Ảnh nhị phân",
                     use_container_width=True)
        with col4:
            st.markdown("**Làm sạch nhiễu**")
            st.image(plate_steps["cleaned"], caption="Sau Morphology Open",
                     use_container_width=True)

        # Segmentation result
        st.markdown("**Character Segmentation (Phân tách ký tự)**")
        st.image(
            plate_steps["segmented"],
            caption=f"Số ký tự phát hiện: {plate_steps['char_count']}",
            use_container_width=True,
        )
        st.caption("Các hộp màu đánh dấu từng ký tự được tách ra")

        # 28×28 character grid
        _display_char_images(plate_steps.get("char_images_28x28", []))


def _display_char_images(char_images: list) -> None:
    """Show a grid of 28×28 character images (upscaled for visibility)."""
    if not char_images:
        return

    st.markdown("**📦 Ký tự 28×28 (chuẩn hóa)**")
    st.caption(
        "Mỗi ký tự được resize về 28×28 pixel với padding giữ tỉ lệ "
        "- sẵn sàng cho OCR/ML"
    )

    num_chars = len(char_images)
    cols_per_row = min(10, num_chars)

    cols = st.columns(cols_per_row)
    for i, char_img in enumerate(char_images[:10]):
        with cols[i % cols_per_row]:
            display_img = cv2.resize(char_img, (112, 112),
                                     interpolation=cv2.INTER_NEAREST)
            st.image(display_img, caption=f"#{i+1}", use_container_width=True)

    if num_chars > 10:
        cols2 = st.columns(min(10, num_chars - 10))
        for i, char_img in enumerate(char_images[10:20]):
            with cols2[i]:
                display_img = cv2.resize(char_img, (112, 112),
                                         interpolation=cv2.INTER_NEAREST)
                st.image(display_img, caption=f"#{i+11}", use_container_width=True)


# ---------------------------------------------------------------------------
# Recognition result rendering
# ---------------------------------------------------------------------------

def display_plate_result(plate, plate_roi_rgb: np.ndarray) -> None:
    """Render a single plate's ROI image + OCR result side-by-side."""

    col_plate, col_result = st.columns([1, 2])

    with col_plate:
        st.image(plate_roi_rgb, caption="Vùng biển số đã cắt",
                 use_container_width=True)
        if plate.corrected_image is not None:
            st.image(plate.corrected_image,
                     caption="Sau Perspective Correction",
                     use_container_width=True)

    with col_result:
        st.markdown("#### Kết quả Nhận dạng")

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

        st.metric(
            label="Độ tin cậy",
            value=f"{plate.confidence:.1f}%",
            delta=(
                "Cao" if plate.confidence > 80
                else ("Trung bình" if plate.confidence > 50 else "Thấp")
            ),
        )

        st.markdown(f"""
        | Thuộc tính | Giá trị |
        |----------|-------|
        | **Loại biển số** | {plate.plate_type} |
        | **Phương pháp phát hiện** | {plate.detection_method} |
        | **Bounding Box** | {plate.box} |
        """)


def display_detection_overlay(image: np.ndarray, plates: list) -> None:
    """Draw all detected plates on *image* and display the result."""

    st.subheader("📍 Trực quan hóa Phát hiện")
    result_img = image.copy()

    for plate in plates:
        x, y, w, h = plate.box
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label = f"{plate.text} ({plate.confidence:.0f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(result_img, (x, y - th - 10), (x + tw + 10, y),
                      (0, 255, 0), -1)
        cv2.putText(result_img, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_img_rgb, caption="Kết quả Phát hiện", use_container_width=True)


def display_no_detection_tips() -> None:
    """Show tips when no plate was found."""
    st.warning("⚠️ Không phát hiện biển số xe trong ảnh này.")
    st.markdown("""
    **Mẹo để phát hiện tốt hơn:**
    - Đảm bảo biển số rõ ràng và không quá nhỏ
    - Thử bật "Chế độ Khó" cho ảnh khó
    - Đảm bảo ảnh có ánh sáng tốt
    - Biển số không nên bị nghiêng nhiều hoặc che khuất
    """)


def display_sample_images(sample_folder) -> None:
    """Show a row of sample thumbnail images."""
    if not sample_folder.exists():
        return

    sample_images = sorted(sample_folder.glob("*.jpg"))[:5]
    if not sample_images:
        return

    cols = st.columns(5)
    for i, img_path in enumerate(sample_images):
        with cols[i]:
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_small = cv2.resize(img_rgb, (150, 100))
                st.image(img_small, caption=img_path.name[:15] + "...")
