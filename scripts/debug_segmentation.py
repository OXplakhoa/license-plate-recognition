"""Debug script to analyze why segmentation misses characters."""
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

# Show all connected components before filtering
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
print(f'Total connected components: {num_labels - 1}')
print(f'ROI area: {corrected.shape[0] * corrected.shape[1]}')
print()

# Sort by area and show details
comps = []
for i in range(1, num_labels):
    x1, y1, w1, h1, area = stats[i]
    comps.append((i, x1, y1, w1, h1, area))

comps.sort(key=lambda c: c[5], reverse=True)  # Sort by area

print('Top 20 components (sorted by area):')
for i, (idx, x1, y1, w1, h1, area) in enumerate(comps[:20]):
    ar = w1 / h1 if h1 > 0 else 0
    height_ratio = h1 / corrected.shape[0]
    area_ratio = area / (corrected.shape[0] * corrected.shape[1])
    print(f'  #{i+1}: pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={height_ratio:.2f} Area={area:4d} ({area_ratio*100:.3f}%)')

# Now segment
seg_result = segment_characters(corrected, plate_type='bike', debug=True)
print(f"\nSegmentation found: {len(seg_result.boxes)} chars")
print(f'Debug info: {seg_result.debug_info}')
print()
print('Final boxes:')
for i, box in enumerate(seg_result.boxes):
    print(f'  Char {i+1}: {box}')

# Save visualization
debug_img = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
for box in seg_result.boxes:
    cv2.rectangle(debug_img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 1)

# Also mark top 10 components in red
for i, (idx, x1, y1, w1, h1, area) in enumerate(comps[:10]):
    cv2.rectangle(debug_img, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 1)
    cv2.putText(debug_img, str(i+1), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

cv2.imwrite(r'F:\CODE\XLA\debug_segmentation.png', debug_img)
print(f"\n✓ Saved debug image: F:\\CODE\\XLA\\debug_segmentation.png")

# Save binary image too
cv2.imwrite(r'F:\CODE\XLA\debug_binary.png', inverted)
print(f"✓ Saved binary image: F:\\CODE\\XLA\\debug_binary.png")
