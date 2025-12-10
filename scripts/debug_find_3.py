"""Debug script to find missing character '3'."""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from src.character_segmenter import segment_characters, binarize_plate
from src.lp_detector import detect_with_character_validation, correct_plate_perspective_and_skew

# Load image
img = cv2.imread(r'F:\CODE\XLA\data\test_images\CarTGMT\AQUA7_91177_checkin_2020-10-27-9-51IC0IB7Un_5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect
dets, _ = detect_with_character_validation(gray, min_char_score=0.35, debug=True)
x, y, w, h = dets[0]['box']
roi = gray[y:y+h, x:x+w]

# Correct
roi_box = (0, 0, roi.shape[1], roi.shape[0])
corrected, _ = correct_plate_perspective_and_skew(roi, roi=roi_box, deskew=True)
if corrected is None:
    corrected = roi

print(f"Corrected ROI size: {corrected.shape}")

# Binarize  
binary, inverted = binarize_plate(corrected, method='combined', denoise=True, aggressive=False)

# Check top-left corner (where first char "3" should be)
print("\n=== Analyzing top-left region (where '3' should be) ===")
# First char of bike plate should be around x=70-90 (left area of top line)
# Based on other chars at x=108, 124, first one should be around x=90 or less

# Look for components in top-left area
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

print(f"\nComponents in top row (y < 45):")
for i in range(1, num_labels):
    x1, y1, w1, h1, area = stats[i]
    if y1 < 45:  # Top half
        ar = w1 / h1 if h1 > 0 else 0
        print(f"  #{i}: pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} Area={area}")

# Now check what's in the grayscale image in the expected location
# Character at x=108 is likely "0", at x=124 is "E", so "3" should be around x=90
print("\n=== Checking grayscale values in expected first char location ===")
# Expected x location: if spacing is ~16px, then x = 108 - 16 = 92
print(f"Grayscale slice [10:45, 85:110]:")
print(corrected[10:45, 85:110])

# Check if first char is touching the edge
print("\n=== Left edge analysis ===")
print(f"Leftmost columns of binary (inverted):")
print(inverted[10:45, :20])

# Maybe binarization lost the char?
print("\n=== Alternative binarization test ===")
# Try Otsu directly
_, otsu = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_inv = cv2.bitwise_not(otsu)
num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(otsu_inv, connectivity=8)
print(f"Otsu components in top row (y < 45):")
for i in range(1, num_labels2):
    x1, y1, w1, h1, area = stats2[i]
    if y1 < 45 and area > 50:  # Top half, min area
        ar = w1 / h1 if h1 > 0 else 0
        print(f"  #{i}: pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} Area={area}")

cv2.imwrite(r'F:\CODE\XLA\debug_otsu.png', otsu_inv)
print(f"\n✓ Saved Otsu binary: F:\\CODE\\XLA\\debug_otsu.png")
