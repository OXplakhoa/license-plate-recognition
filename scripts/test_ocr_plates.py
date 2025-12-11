#!/usr/bin/env python
"""Test OCR on all candidate plates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import pytesseract
from src.lp_detector import detect_multi_preset
from src.ocr_engine import configure_tesseract, ocr_plate_line
from src.heuristics import apply_heuristics, is_valid_plate

configure_tesseract()

img_path = 'data/test_images/CarTGMT/AQUA7_51443_checkoutex_2020-10-22-9-48a3Jfs9hifV.jpg'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Focus on plate 2 which is the actual license plate
print("Testing various OCR approaches on the license plate region (219, 428, 128, 22):")
x, y, w, h = 219, 428, 128, 22
roi = gray[y:y+h, x:x+w]

print(f"Original ROI shape: {roi.shape}")

# Save ROI for visual inspection
cv2.imwrite('debug_output/plate_roi_original.png', roi)

# Upscale
scale = 4
roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
print(f"Upscaled ROI shape: {roi_large.shape}")
cv2.imwrite('debug_output/plate_roi_upscaled.png', roi_large)

# Try different preprocessing
# 1. CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
roi_clahe = clahe.apply(roi_large)
cv2.imwrite('debug_output/plate_roi_clahe.png', roi_clahe)

# 2. Otsu thresholding
_, roi_otsu = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('debug_output/plate_roi_otsu.png', roi_otsu)

# 3. Adaptive thresholding
roi_adapt = cv2.adaptiveThreshold(roi_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('debug_output/plate_roi_adaptive.png', roi_adapt)

# 4. Inverted Otsu
roi_otsu_inv = cv2.bitwise_not(roi_otsu)
cv2.imwrite('debug_output/plate_roi_otsu_inv.png', roi_otsu_inv)

print("\nTesting Tesseract with different images and configs:")

configs = [
    ('--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'default'),
    ('--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'single word'),
    ('--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'uniform block'),
    ('--psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'raw line'),
]

test_images = [
    ('upscaled', roi_large),
    ('clahe', roi_clahe),
    ('otsu', roi_otsu),
    ('otsu_inv', roi_otsu_inv),
    ('adaptive', roi_adapt),
]

for img_name, test_img in test_images:
    print(f"\n  {img_name}:")
    for config, config_name in configs:
        try:
            text = pytesseract.image_to_string(test_img, config=config).strip()
            if text:
                print(f"    {config_name}: '{text}'")
        except Exception as e:
            print(f"    {config_name}: ERROR - {e}")
