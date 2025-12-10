"""Visualize the full detection process with full plate view."""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from src.lp_detector import detect_with_character_validation, correct_plate_perspective_and_skew
from src.character_segmenter import segment_characters, binarize_plate

# Load image
img = cv2.imread(r'F:\CODE\XLA\data\test_images\CarTGMT\AQUA7_91177_checkin_2020-10-27-9-51IC0IB7Un_5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect with more padding
dets, _ = detect_with_character_validation(gray, min_char_score=0.35, debug=True)
x, y, w, h = dets[0]['box']

# Add more padding on left to capture full plate
pad_left = 50  # Extra padding on left
pad_other = 10

x1 = max(0, x - pad_left)
y1 = max(0, y - pad_other)
x2 = min(gray.shape[1], x + w + pad_other)
y2 = min(gray.shape[0], y + h + pad_other)

roi_padded = gray[y1:y2, x1:x2]

print(f"Padded ROI size: {roi_padded.shape}")

# Correct perspective on padded ROI  
roi_box = (0, 0, roi_padded.shape[1], roi_padded.shape[0])
corrected, _ = correct_plate_perspective_and_skew(roi_padded, roi=roi_box, deskew=True)
if corrected is None:
    corrected = roi_padded

print(f"Corrected size: {corrected.shape}")

# Binarize
binary, inverted = binarize_plate(corrected, method='combined', denoise=True, aggressive=False)

# Find components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

print(f'\nComponents with character-like shape (AR 0.2-0.8, H% 0.15-0.5):')
char_comps = []
for i in range(1, num_labels):
    x1, y1, w1, h1, area = stats[i]
    if area < 50:
        continue
    ar = w1 / h1 if h1 > 0 else 0
    height_ratio = h1 / corrected.shape[0]
    
    # Character-like constraints
    if 0.15 < ar < 0.8 and 0.15 < height_ratio < 0.5:
        char_comps.append((x1, y1, w1, h1, area, ar, height_ratio))
        print(f'  pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={height_ratio:.2f} Area={area}')

print(f'\nTotal character-like components: {len(char_comps)}')

# Draw on image
debug_img = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
for x1, y1, w1, h1, area, ar, hr in char_comps:
    cv2.rectangle(debug_img, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 1)

cv2.imwrite(r'F:\CODE\XLA\debug_padded_chars.png', debug_img)
cv2.imwrite(r'F:\CODE\XLA\debug_padded_binary.png', inverted)
cv2.imwrite(r'F:\CODE\XLA\debug_padded_corrected.png', corrected)

print(f"\n✓ Saved padded corrected: F:\\CODE\\XLA\\debug_padded_corrected.png")
print(f"✓ Saved padded binary: F:\\CODE\\XLA\\debug_padded_binary.png")
print(f"✓ Saved padded chars: F:\\CODE\\XLA\\debug_padded_chars.png")
