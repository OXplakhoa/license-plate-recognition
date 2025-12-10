"""Test dual preprocessing mode in segment_characters."""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from src.character_segmenter import segment_characters, binarize_plate
from src.lp_detector import detect_with_character_validation

# Load image  
img = cv2.imread(r'F:\CODE\XLA\data\test_images\CarTGMT\AQUA7_91177_checkin_2020-10-27-9-51IC0IB7Un_5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect
dets, _ = detect_with_character_validation(gray, min_char_score=0.35, debug=True)
x, y, w, h = dets[0]['box']
roi = gray[y:y+h, x:x+w]

print(f"ROI size: {roi.shape}")

# Test segment_characters with debug to see which mode is chosen
print("\n=== Testing segment_characters() ===")
seg_result = segment_characters(roi, plate_type='bike', debug=True)
print(f"Result: {len(seg_result.boxes)} characters")
print(f"Debug info: {seg_result.debug_info}")

# Manually check light vs heavy
print("\n=== Manual comparison ===")
# Light mode
binary_light, inverted_light = binarize_plate(roi, method='combined', denoise=False, light_preprocess=True)
num_l, _, stats_l, _ = cv2.connectedComponentsWithStats(inverted_light, connectivity=8)
chars_light = 0
for i in range(1, num_l):
    x1, y1, w1, h1, area = stats_l[i]
    if area < 50:
        continue
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / roi.shape[0]
    if 0.10 < ar < 1.5 and 0.15 < hr < 0.5:
        chars_light += 1

# Heavy mode
binary_heavy, inverted_heavy = binarize_plate(roi, method='combined', denoise=True, light_preprocess=False)
num_h, _, stats_h, _ = cv2.connectedComponentsWithStats(inverted_heavy, connectivity=8)
chars_heavy = 0
for i in range(1, num_h):
    x1, y1, w1, h1, area = stats_h[i]
    if area < 50:
        continue
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / roi.shape[0]
    if 0.10 < ar < 1.5 and 0.15 < hr < 0.5:
        chars_heavy += 1

print(f"Light mode chars: {chars_light}")
print(f"Heavy mode chars: {chars_heavy}")

# Save both
cv2.imwrite(r'F:\CODE\XLA\debug_dual_light.png', inverted_light)
cv2.imwrite(r'F:\CODE\XLA\debug_dual_heavy.png', inverted_heavy)
print("\n✓ Saved both modes")

# Check what extract_boxes_from_binary would return
print("\n=== Simulating extract_boxes_from_binary ===")
from src.character_segmenter import _filter_components

h_roi, w_roi = roi.shape[:2]
area_roi = h_roi * w_roi
min_area = area_roi * 0.002
max_area = area_roi * 0.20
ar_range = (0.10, 1.5)
min_height_ratio = 0.15

# Light
mask_l = (inverted_light > 0).astype("uint8")
boxes_l = _filter_components(mask_l, min_area=min_area, max_area=max_area, ar_range=ar_range, min_height_ratio=min_height_ratio)
print(f"Light - after _filter_components: {len(boxes_l)} boxes")

# Heavy
mask_h = (inverted_heavy > 0).astype("uint8")
boxes_h = _filter_components(mask_h, min_area=min_area, max_area=max_area, ar_range=ar_range, min_height_ratio=min_height_ratio)
print(f"Heavy - after _filter_components: {len(boxes_h)} boxes")

# Apply additional filtering
def additional_filter(boxes, w_roi):
    if not boxes:
        return []
    heights = [b[3] for b in boxes]
    areas = [b[2] * b[3] for b in boxes]
    median_h = np.median(heights)
    median_area = np.median(areas)
    
    margin = 2
    filtered = []
    for b in boxes:
        bx, by, bw, bh = b
        box_area = bw * bh
        if not (0.4 * median_h <= bh <= 1.6 * median_h):
            continue
        if box_area < 0.2 * median_area:
            continue
        if bx <= margin or bx + bw >= w_roi - margin:
            continue
        filtered.append(b)
    return filtered

filtered_l = additional_filter(boxes_l, w_roi)
filtered_h = additional_filter(boxes_h, w_roi)
print(f"Light - after additional filter: {len(filtered_l)} boxes")
print(f"Heavy - after additional filter: {len(filtered_h)} boxes")

# Show which would be chosen
if len(filtered_l) > len(filtered_h) and len(filtered_l) <= 10:
    print("\n→ Would choose LIGHT mode")
else:
    print("\n→ Would choose HEAVY mode")
