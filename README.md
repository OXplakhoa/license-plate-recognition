# Vietnamese License Plate Recognition

**Dự án nhận dạng biển số xe sử dụng các kỹ thuật xử lý ảnh truyền thống (không dùng Deep Learning)**

## 📋 Mô Tả

Hệ thống nhận dạng biển số xe Việt Nam sử dụng:
- **Xử lý ảnh số truyền thống:** Edge detection, morphology, contour analysis
- **OCR:** Tesseract OCR cho nhận dạng ký tự
- **Không dùng CNN/YOLO** - Phù hợp với môn Xử Lý Ảnh

## 🎯 Pipeline

```
Input Image
    ↓
[1] Tiền xử lý (Preprocessing)
    - Grayscale, Blur, Median Filter, CLAHE
    ↓
[2] Phát hiện biển số (Detection)
    - Canny edge, Morphology, Contour filtering
    ↓
[3] Phân tách ký tự (Segmentation)
    - Otsu threshold, Character extraction
    ↓
[4] Nhận dạng (Recognition)
    - Tesseract OCR, Validation
    ↓
Output: Biển số xe (12A-345.67)
```

## 🚀 Bắt Đầu Nhanh

### 1. Cài Đặt
```bash
# Xem SETUP.md để chi tiết
pip install -r requirements.txt
```

### 2. Phát Triển
```bash
# Sử dụng Jupyter Notebook
jupyter notebook
# Mở: notebooks/lp_recognition_exploration.ipynb
```

### 3. Chạy
```bash
# Thử tự động (car 2 dòng, car 1 dòng, bike)
python main.py --image data/test_images/sample.jpg --plate-type auto

# Ép kiểu biển (vd: biển chữ nhật 1 dòng)
python main.py --image data/test_images/sample.jpg --plate-type car1
```

## 📁 Cấu Trúc Thư Mục

```
lp_recognition/
├── notebooks/
│   └── lp_recognition_exploration.ipynb    # Phát triển pipeline
├── src/
│   ├── __init__.py
│   ├── lp_detector.py                      # Phát hiện biển số
│   ├── character_segmenter.py              # Phân tách ký tự
│   ├── ocr_engine.py                       # Tesseract wrapper
│   ├── preprocess.py                       # Tiền xử lý ảnh
│   ├── pipeline.py                         # Chuỗi tiền xử lý → detect → segment → OCR
│   └── utils.py                            # Các hàm tiện ích
├── tests/
│   ├── test_detector.py
│   └── test_segmenter.py
├── data/
│   ├── test_images/                        # Ảnh test
│   └── ground_truth.txt                    # Expected biển số
├── main.py                                 # Entry point
├── requirements.txt
├── SETUP.md                                # Hướng dẫn cài đặt
├── .env.example                            # Cấu hình Tesseract
└── README.md
```

## 🔧 Công Nghệ

| Component | Công Cụ | Phiên Bản |
|-----------|---------|----------|
| Image Processing | OpenCV | 4.8+ |
| Numerical | NumPy | 1.21+ |
| OCR | Tesseract | 5.x |
| Visualization | Matplotlib | 3.5+ |
| Development | Jupyter | 1.0+ |

## 📝 Các Bước Thực Hiện

### Phase 1: Khám Phá & Phát Triển (Jupyter Notebook)
- [ ] Bước 1: Tiền xử lý ảnh
- [ ] Bước 2-3: Phát hiện biển số
- [ ] Bước 4: Phân tách ký tự
- [ ] Bước 5: Nhận dạng OCR
- [ ] Bước 6: Hiển thị kết quả

### Phase 2: Refactoring & Production (Python Modules)
- [ ] Tách code thành modules
- [ ] Thêm unit tests
- [ ] Tối ưu performance

## 🎓 Yêu Cầu Dự Án

✅ Áp dụng kỹ thuật xử lý ảnh số & OCR  
✅ Phát hiện, tách, nhận dạng biển số từ ảnh thực tế  
✅ **KHÔNG dùng CNN/Deep Learning**  
✅ Hỗ trợ ảnh .jpg, .png, camera  
✅ Kiểm thử từng bước xử lý  

## 🧪 Kiểm Thử

```bash
# Chạy unit tests
pytest tests/

# Kiểm tra coverage
pytest --cov=src tests/
```

## 📊 Tiêu Chí Thành Công

- [ ] Phát hiện biển số với độ chính xác > 90%
- [ ] Phân tách ký tự chính xác
- [ ] Nhận dạng biển số khớp ground truth
- [ ] Xử lý nhiều điều kiện ánh sáng & góc
- [ ] Runtime < 1 giây/ảnh
- [ ] Code tái sử dụng, có test

## 📚 Tài Liệu Tham Khảo

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Guide](https://numpy.org/doc/)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)

## 🤝 Đóng Góp

Các đề xuất cải tiến: Tạo issue hoặc pull request!

## 📄 Giấy Phép

MIT License - Xem LICENSE file

---

**Lưu ý:** Dự án này được thực hiện cho môn "Xử Lý Ảnh". Sử dụng các kỹ thuật truyền thống thay vì Deep Learning để hiểu rõ hơn về Image Processing.
