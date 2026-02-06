"""
Image-input helpers for the Streamlit demo application.

Handles the three input sources (file upload, clipboard/URL, camera)
and returns a BGR ``np.ndarray`` or ``None``.
"""

import base64
import urllib.request

import cv2
import numpy as np
import streamlit as st


def get_input_image(input_source: str) -> np.ndarray | None:
    """
    Display the appropriate Streamlit input widget and return the
    decoded BGR image, or *None* if no image was provided yet.
    """
    if input_source == "Tải ảnh lên (JPG/PNG)":
        return _from_file_upload()
    elif input_source == "Dán từ Clipboard":
        return _from_clipboard()
    else:
        return _from_camera()


# ---------------------------------------------------------------------------
# Private helpers for each input mode
# ---------------------------------------------------------------------------

def _from_file_upload() -> np.ndarray | None:
    uploaded_file = st.file_uploader(
        "Tải lên ảnh chứa xe có biển số",
        type=["jpg", "jpeg", "png"],
        help="Định dạng hỗ trợ: JPG, JPEG, PNG",
    )
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def _from_clipboard() -> np.ndarray | None:
    st.info("📋 Dán ảnh Base64 hoặc URL vào ô bên dưới")

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

    clipboard_input = st.text_area(
        "Dán Base64 hoặc URL ảnh:",
        height=120,
        placeholder=(
            "Dán vào đây:\n"
            "• data:image/png;base64,iVBORw0KGgo...\n"
            "• https://example.com/car.jpg"
        ),
        key="clipboard_input",
    )

    if not clipboard_input or not clipboard_input.strip():
        return None

    clipboard_input = clipboard_input.strip()

    try:
        image = _decode_clipboard(clipboard_input)
        if image is not None:
            st.success("✅ Đã tải ảnh thành công!")
        else:
            st.error("❌ Không thể đọc ảnh. Vui lòng kiểm tra lại dữ liệu.")
        return image
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý ảnh: {e}")
        return None


def _decode_clipboard(text: str) -> np.ndarray | None:
    """Attempt to decode *text* as base64-data-URI, plain URL, or raw base64."""
    if text.startswith("data:image"):
        raw = text.split(",", 1)[1] if "," in text else text
        return _bytes_to_image(base64.b64decode(raw))

    if text.startswith("http"):
        req = urllib.request.Request(text, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return _bytes_to_image(resp.read())

    # Last resort – try raw base64
    try:
        return _bytes_to_image(base64.b64decode(text))
    except Exception:
        st.error("❌ Định dạng không hợp lệ. Vui lòng dán URL hoặc Base64.")
        return None


def _bytes_to_image(raw_bytes: bytes) -> np.ndarray | None:
    nparr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _from_camera() -> np.ndarray | None:
    st.warning("📹 Đầu vào Camera - Nhấn 'Chụp ảnh' để chụp")
    camera_input = st.camera_input("Chụp ảnh từ camera")
    if camera_input is None:
        return None
    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
