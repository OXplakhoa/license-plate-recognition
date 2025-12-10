"""Debug binarize_plate with light_preprocess=True."""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from src.character_segmenter import binarize_plate

# Load ROI
roi = cv2.imread(r'F:\CODE\XLA\debug_no_perspective_roi.png', cv2.IMREAD_GRAYSCALE)
print(f"ROI size: {roi.shape}")

# Call binarize_plate with light_preprocess=True
print("\n=== binarize_plate(light_preprocess=True, method='combined') ===")
binary, inverted = binarize_plate(roi, method='combined', denoise=False, light_preprocess=True)

# Check components
num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

print(f"Total components: {num_labels - 1}")

h, w = roi.shape
area = h * w
min_area = area * 0.002

print("\nCharacters in TOP row (y < 40):")
chars = []
for i in range(1, num_labels):
    x1, y1, w1, h1, comp_area = stats[i]
    if y1 >= 40:
        continue
    if comp_area < 50:
        continue
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / h
    if 0.10 <= ar <= 1.5 and hr >= 0.15:
        chars.append((x1, y1, w1, h1, comp_area))
        print(f"  pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={hr:.2f} Area={comp_area}")

print(f"\nTotal top row chars: {len(chars)}")

cv2.imwrite(r'F:\CODE\XLA\debug_binarize_light.png', inverted)
print(f"\n✓ Saved: F:\\CODE\\XLA\\debug_binarize_light.png")

# Also check manual adaptive
print("\n=== Manual adaptive on same blurred image ===")
blurred = cv2.GaussianBlur(roi, (3, 3), 0)
adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

num_labels2, _, stats2, _ = cv2.connectedComponentsWithStats(adaptive, connectivity=8)
print(f"Total components: {num_labels2 - 1}")

chars2 = []
for i in range(1, num_labels2):
    x1, y1, w1, h1, comp_area = stats2[i]
    if y1 >= 40:
        continue
    if comp_area < 50:
        continue
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / h
    if 0.10 <= ar <= 1.5 and hr >= 0.15:
        chars2.append((x1, y1, w1, h1, comp_area))
        print(f"  pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={hr:.2f} Area={comp_area}")

print(f"\nTotal top row chars: {len(chars2)}")

cv2.imwrite(r'F:\CODE\XLA\debug_manual_adaptive.png', adaptive)
