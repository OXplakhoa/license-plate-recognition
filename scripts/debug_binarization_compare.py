"""Analyze why first characters are missing - check for merging with border."""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from src.character_segmenter import binarize_plate

# Load ROI
roi = cv2.imread(r'F:\CODE\XLA\debug_no_perspective_roi.png', cv2.IMREAD_GRAYSCALE)
print(f"ROI size: {roi.shape}")

# Try pure adaptive threshold (best result from earlier tests)
adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

print("\n=== Adaptive (15,5) - No preprocessing ===")
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(adaptive, connectivity=8)

chars = []
for i in range(1, num_labels):
    x1, y1, w1, h1, area = stats[i]
    if area < 50:
        continue
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / roi.shape[0]
    if 0.15 < ar < 0.8 and 0.2 < hr < 0.5:
        chars.append((x1, y1, w1, h1, area, ar, hr))

chars.sort(key=lambda c: (c[1], c[0]))  # Sort by y, then x
print(f"Found {len(chars)} character-like components:")
for x1, y1, w1, h1, area, ar, hr in chars:
    print(f"  pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={hr:.2f}")

cv2.imwrite(r'F:\CODE\XLA\debug_pure_adaptive.png', adaptive)

# Now try with our binarize_plate function
print("\n=== binarize_plate (combined) - With preprocessing ===")
binary, inverted = binarize_plate(roi, method='combined', denoise=True, aggressive=False)

num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

chars2 = []
for i in range(1, num_labels2):
    x1, y1, w1, h1, area = stats2[i]
    if area < 50:
        continue
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / roi.shape[0]
    if 0.15 < ar < 0.8 and 0.2 < hr < 0.5:
        chars2.append((x1, y1, w1, h1, area, ar, hr))

chars2.sort(key=lambda c: (c[1], c[0]))
print(f"Found {len(chars2)} character-like components:")
for x1, y1, w1, h1, area, ar, hr in chars2:
    print(f"  pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={hr:.2f}")

# Key difference: the preprocessing (CLAHE, denoise, sharpen, blur) may be removing some characters
# Let's compare side by side
print("\n=== Comparison ===")
print(f"Pure adaptive: {len(chars)} chars")
print(f"binarize_plate: {len(chars2)} chars")

# Check what binarize_plate is actually selecting
# The 'combined' method should try adaptive(15,5), adaptive(11,3), and otsu
# and pick the best one

# Let's manually check the score for each method
from src.character_segmenter import _count_char_like_components

# First preprocess like binarize_plate does
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
denoised = cv2.fastNlMeansDenoising(clahe, None, h=5, templateWindowSize=7, searchWindowSize=21)
kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)

# Try each method
_, otsu_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adapt_bin1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
adapt_bin2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

otsu_score = _count_char_like_components(cv2.bitwise_not(otsu_bin))
adapt1_score = _count_char_like_components(cv2.bitwise_not(adapt_bin1))
adapt2_score = _count_char_like_components(cv2.bitwise_not(adapt_bin2))

print(f"\n=== Method scores (with preprocessing) ===")
print(f"Otsu score: {otsu_score}")
print(f"Adaptive(15,5) score: {adapt1_score}")
print(f"Adaptive(11,3) score: {adapt2_score}")
print(f"Best: {'Otsu' if otsu_score >= max(adapt1_score, adapt2_score) else 'Adaptive(15,5)' if adapt1_score >= adapt2_score else 'Adaptive(11,3)'}")

# Save all versions
cv2.imwrite(r'F:\CODE\XLA\debug_otsu_preprocessed.png', cv2.bitwise_not(otsu_bin))
cv2.imwrite(r'F:\CODE\XLA\debug_adapt15_preprocessed.png', cv2.bitwise_not(adapt_bin1))
cv2.imwrite(r'F:\CODE\XLA\debug_adapt11_preprocessed.png', cv2.bitwise_not(adapt_bin2))
print("\n✓ Saved all binarization variants")
