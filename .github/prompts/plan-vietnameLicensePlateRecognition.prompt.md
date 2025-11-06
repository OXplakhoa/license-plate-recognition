# Plan: Vietnamese License Plate Recognition - Traditional Image Processing

**TL;DR:** Xây dựng hệ thống nhận dạng biển số xe bằng các kỹ thuật xử lý ảnh truyền thống (không dùng CNN/YOLO). Sử dụng Jupyter Notebook cho giai đoạn phát triển & kiểm thử interactively, sau đó tái cấu trúc thành Python modules. Pipeline gồm 6 bước chính với kiểm thử trực quan từng bước.

## Implementation Steps

### 1. Thiết lập cấu trúc dự án và môi trường
- Tạo thư mục: `notebooks/`, `src/`, `tests/`, `data/` (train/test images)
- Cài đặt dependencies: OpenCV, NumPy, Tesseract, pytesseract, matplotlib, Pillow
- Cấu hình Tesseract đường dẫn trên Windows

### 2. Tạo Jupyter Notebook cho pipeline khám phá
- File: `notebooks/lp_recognition_exploration.ipynb`
- Mục đích: Phát triển & kiểm thử interactively từng bước xử lý
- Cấu trúc: 6 cell tương ứng 6 bước trong `project.txt` + cell tóm tắt

### 3. Bước 1: Nhập & tiền xử lý ảnh
- Hàm: `load_and_preprocess()` trong cell Notebook
- Xử lý: Grayscale → Gaussian Blur → Median Filter → CLAHE contrast enhancement
- **Kiểm thử:** Hiển thị song song: ảnh gốc vs ảnh sau mỗi giai đoạn tiền xử lý

### 4. Bước 2-3: Phát hiện vùng biển số
- Hàm: `detect_license_plate()` 
- Xử lý: Canny edge → Morphological close/open → Find contours → Filter by aspect ratio (4:1-5:1)
- **Kiểm thử:** Vẽ contour, bounding box biển số trên ảnh gốc; in kích thước & tỷ lệ phát hiện

### 5. Bước 4: Chuẩn hóa & phân tách ký tự
- Hàm: `segment_characters()`
- Xử lý: Otsu threshold → Find character contours → Sort L-to-R → Resize 28×28
- **Kiểm thử:** Hiển thị grid các ký tự được tách riêng; kiểm tra số lượng ký tự

### 6. Bước 5: Nhận dạng ký tự với Tesseract OCR
- Hàm: `recognize_plate()` (wrapper Tesseract)
- Xử lý: PSM mode 6 cho từng ký tự → Validation format VN (12A-345.67)
- **Kiểm thử:** In từng ký tự nhận diện + confidence; kết quả biển số cuối cùng

### 7. Bước 6: Hiển thị & đánh giá kết quả tổng hợp
- Visualize: Ảnh gốc + biển số được detect + kết quả OCR final
- So sánh: Nếu có ground truth, tính accuracy

### 8. Tái cấu trúc thành Python modules
- Tách Notebook → `src/lp_detector.py`, `src/character_segmenter.py`, `src/ocr_engine.py`
- Tạo `main.py` entry point (hỗ trợ file/camera input)
- Thêm unit tests trong `tests/`

### 9. Tối ưu & mở rộng
- Điều chỉnh tham số (threshold, kernel size) dựa trên kết quả kiểm thử
- Hỗ trợ video stream / batch processing
- Xử lý edge cases (biển số bị nghiêng, ánh sáng yếu)

## Key Considerations

### Jupyter vs Python thuần?
- **Khuyến cáo:** Jupyter cho phát triển (cell-by-cell testing, trực quan hình ảnh), Python thuần cho production
- Sau khi hoàn thiện pipeline trong Notebook, tái cấu trúc thành modules `.py` có thể tái sử dụng

### Dữ liệu kiểm thử?
- Chuẩn bị 10-20 ảnh xe thực tế với điều kiện khác nhau (ánh sáng, góc chụp, khoảng cách)
- Lưu ground truth biển số để đánh giá accuracy OCR

### Thứ tự ưu tiên nếu gặp khó khăn?
- **Phát hiện LP:** Dùng Sobel + morphology, nếu không được thay Canny
- **Phân tách ký tự:** Projection-based (đơn giản) trước, nếu không được thử contour-based
- **OCR:** Tesseract PSM 6 (ký tự đơn) đơn giản nhất, nếu không chính xác thử PSM modes khác

## Architecture Overview

```
project/
├── notebooks/
│   └── lp_recognition_exploration.ipynb  (development & testing)
├── src/
│   ├── lp_detector.py       (plate detection logic)
│   ├── character_segmenter.py (character segmentation)
│   ├── ocr_engine.py        (Tesseract wrapper & recognition)
│   └── utils.py             (preprocessing utilities)
├── tests/
│   ├── test_detector.py
│   └── test_segmenter.py
├── data/
│   ├── test_images/         (sample license plate images)
│   └── ground_truth.txt     (expected plate numbers)
├── main.py                  (entry point for production use)
└── requirements.txt
```

## Technology Stack

| Component | Tool/Library | Purpose |
|-----------|---|---|
| Image Processing | OpenCV (cv2) | Grayscale, filters, edge detection, morphology, contours |
| Numerical Ops | NumPy | Array manipulation, matrix operations |
| OCR | Tesseract + pytesseract | Character recognition |
| Visualization | Matplotlib | Display results (for testing) |
| Development | Jupyter Notebook | Interactive pipeline development |
| Image I/O | Pillow | Alternative image handling |

## Workflow

```
Input Image
    ↓
[Preprocessing] → Grayscale, Blur, Median, CLAHE
    ↓ (Display results)
[License Plate Detection] → Canny, Morphology, Contour filtering
    ↓ (Display detected region)
[Character Segmentation] → Otsu Threshold, Contour extraction
    ↓ (Display segmented characters)
[OCR Recognition] → Tesseract PSM 6
    ↓ (Display recognized text)
[Validation & Output] → Format check, final plate number
    ↓
Output: Vietnamese License Plate Format (12A-345.67)
```

## Success Criteria

- [ ] Detect license plate region with >90% accuracy on test images
- [ ] Segment individual characters correctly
- [ ] Recognize plate number matching ground truth (if available)
- [ ] Handle various lighting conditions and angles
- [ ] Runtime < 1 second per image on standard CPU
- [ ] Code refactored into reusable modules with tests
