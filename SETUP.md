# Hướng Dẫn Cài Đặt & Thiết Lập Dự Án

## 1. Yêu Cầu Hệ Thống

- **Python:** 3.11 hoặc 3.12 (khuyến cáo) - Tránh Python 3.14 vì chưa có đủ wheel pre-built
- **Windows:** 64-bit
- **Tesseract OCR:** Cần cài đặt riêng

## 2. Cài Đặt Python

### Option A: Sử dụng Python 3.12 (Khuyến cáo)
1. Tải Python 3.12 từ https://www.python.org/downloads/
2. Trong quá trình cài đặt, check "Add Python to PATH"
3. Kiểm tra:
   ```powershell
   python --version
   ```

### Option B: Tải Miniconda (Dễ hơn cho quản lý environments)
1. Tải Miniconda từ https://docs.conda.io/projects/miniconda/
2. Cài đặt với Python 3.12
3. Tạo environment:
   ```powershell
   conda create -n lp-recognition python=3.12
   conda activate lp-recognition
   ```

## 3. Cài Đặt Tesseract OCR

### Windows:
1. Tải installer từ: https://github.com/UB-Mannheim/tesseract/wiki
2. Cài đặt mặc định vào: `C:\Program Files\Tesseract-OCR`
3. Điều chỉnh file `.env` với đường dẫn cài đặt (nếu khác)

## 4. Cài Đặt Python Dependencies

```powershell
# Di chuyển vào thư mục dự án
cd f:\CODE\XLA

# Cài đặt các packages từ requirements.txt
pip install -r requirements.txt

# Cài đặt numpy riêng biệt nếu gặp lỗi
pip install numpy --no-build-isolation
```

## 5. Kiểm Tra Cài Đặt

Chạy script kiểm tra:
```powershell
python -c "
import cv2
import numpy as np
import pytesseract
from PIL import Image
print('✓ OpenCV:', cv2.__version__)
print('✓ NumPy:', np.__version__)
print('✓ Pytesseract: OK')
print('✓ Pillow:', Image.__version__)
"
```

## 6. Cài Đặt Jupyter Notebook (Tùy Chọn)

Nếu cài đặt Jupyter thành công:
```powershell
jupyter notebook
```
Sau đó mở `notebooks/lp_recognition_exploration.ipynb`

## 7. Cấu Trúc Thư Mục

```
f:\CODE\XLA\
├── notebooks/              # Jupyter notebooks cho phát triển
├── src/                    # Python modules
│   ├── __init__.py
│   ├── lp_detector.py
│   ├── character_segmenter.py
│   ├── ocr_engine.py
│   └── utils.py
├── tests/                  # Unit tests
├── data/
│   ├── test_images/        # Ảnh test
│   └── ground_truth.txt    # Biển số expected
├── main.py                 # Entry point
├── requirements.txt
├── .env.example           # Cấu hình Tesseract
└── README.md
```

## 8. Khắc Phục Sự Cố

### Lỗi "No compiler found"
- Cài Python 3.11/3.12 thay vì 3.14
- Hoặc cài Visual C++ Build Tools

### Lỗi "Cannot find tesseract"
- Kiểm tra đường dẫn trong `.env`
- Cài lại Tesseract từ https://github.com/UB-Mannheim/tesseract/wiki

### Lỗi "ModuleNotFoundError: No module named 'cv2'"
```powershell
pip install opencv-python --force-reinstall --no-cache-dir
```

## 9. Các Lệnh Hữu Ích

```powershell
# Xem danh sách packages cài đặt
pip list

# Xóa cache pip (nếu cần)
pip cache purge

# Cài đặt lại tất cả
pip uninstall -r requirements.txt -y; pip install -r requirements.txt

# Chạy tests
python -m pytest tests/

# Chạy ứng dụng main
python main.py --image data/test_images/sample.jpg
```

---

Sau khi hoàn thành các bước này, bạn có thể bắt đầu phát triển dự án!
