"""Analyze plate without perspective correction to see actual characters."""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from src.lp_detector import detect_with_character_validation
from src.character_segmenter import binarize_plate

# Load image  
img = cv2.imread(r'F:\CODE\XLA\data\test_images\CarTGMT\AQUA7_91177_checkin_2020-10-27-9-51IC0IB7Un_5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect
dets, _ = detect_with_character_validation(gray, min_char_score=0.35, debug=True)
x, y, w, h = dets[0]['box']
roi = gray[y:y+h, x:x+w]

print(f"Original ROI size: {roi.shape}")
print(f"Detection box: x={x}, y={y}, w={w}, h={h}")

# Binarize directly without perspective correction
binary, inverted = binarize_plate(roi, method='combined', denoise=True, aggressive=False)

# Find ALL components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

print(f'\nAll components (sorted by area, showing top 20):')
comps = []
for i in range(1, num_labels):
    x1, y1, w1, h1, area = stats[i]
    comps.append((x1, y1, w1, h1, area))

comps.sort(key=lambda c: c[4], reverse=True)

for i, (x1, y1, w1, h1, area) in enumerate(comps[:20]):
    ar = w1 / h1 if h1 > 0 else 0
    height_ratio = h1 / roi.shape[0]
    area_ratio = area / (roi.shape[0] * roi.shape[1])
    print(f'  #{i+1}: pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={height_ratio:.2f} Area%={area_ratio*100:.2f}%')

# Save for visual
debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
for i, (x1, y1, w1, h1, area) in enumerate(comps[:12]):
    color = (0, 255, 0) if area > 100 else (0, 128, 255)
    cv2.rectangle(debug_img, (x1, y1), (x1+w1, y1+h1), color, 1)
    cv2.putText(debug_img, str(i+1), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

cv2.imwrite(r'F:\CODE\XLA\debug_no_perspective.png', debug_img)
cv2.imwrite(r'F:\CODE\XLA\debug_no_perspective_binary.png', inverted)
cv2.imwrite(r'F:\CODE\XLA\debug_no_perspective_roi.png', roi)

print(f"\n✓ Saved ROI: F:\\CODE\\XLA\\debug_no_perspective_roi.png")
print(f"✓ Saved binary: F:\\CODE\\XLA\\debug_no_perspective_binary.png")
print(f"✓ Saved marked: F:\\CODE\\XLA\\debug_no_perspective.png")

# Also try with EasyOCR directly on the ROI
print("\n=== EasyOCR direct on ROI ===")
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    results = reader.readtext(roi, detail=1)
    for bbox, text, conf in results:
        print(f"  Text: '{text}', Conf: {conf:.2%}")
except Exception as e:
    print(f"  EasyOCR error: {e}")
