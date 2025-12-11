import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_engine import ocr_plate_multi_psm
import os
from src.pipeline import LicensePlateRecognizer

# Load real crop
files = [f for f in os.listdir('data/test_images/CarTGMT') if f.endswith('.jpg')][:1]
full_img = cv2.imread(f'data/test_images/CarTGMT/{files[0]}')
r_easy = LicensePlateRecognizer(ocr_engine='easyocr')
res_easy = r_easy.recognize(full_img)
x, y, w, h = res_easy.plates[0].box
crop = full_img[y:y+h, x:x+w]

print(f'Crop shape: {crop.shape}')
print(f'EasyOCR detected: {res_easy.plates[0].text}')

# Test different preprocessing
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# Upscale more
scale = 3.0
gray_large = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
print(f'After 3x upscale: {gray_large.shape}')

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
enhanced = clahe.apply(gray_large)

# Otsu
_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Test OCR
print("\n--- Testing different preprocessing methods ---")
result1 = ocr_plate_multi_psm(enhanced)
print(f'CLAHE only: text="{result1.text}", conf={result1.mean_conf}')

result2 = ocr_plate_multi_psm(binary)
print(f'CLAHE+Otsu: text="{result2.text}", conf={result2.mean_conf}')

# Adaptive
adaptive = cv2.adaptiveThreshold(gray_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
result3 = ocr_plate_multi_psm(adaptive)
print(f'Adaptive: text="{result3.text}", conf={result3.mean_conf}')

# Inverse
_, binary_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
result4 = ocr_plate_multi_psm(binary_inv)
print(f'CLAHE+Otsu INV: text="{result4.text}", conf={result4.mean_conf}')
