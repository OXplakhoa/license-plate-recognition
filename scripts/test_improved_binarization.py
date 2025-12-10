"""Test improved binarization on original ROI (no perspective correction)."""
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

# Test binarization with improved combined method
binary, inverted = binarize_plate(roi, method='combined', denoise=True, aggressive=False)

# Count components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
print(f"\nConnected components: {num_labels - 1}")

# Filter for characters
print("\nCharacter-like components (AR 0.2-0.8, H% 0.2-0.5):")
chars = []
for i in range(1, num_labels):
    x1, y1, w1, h1, area = stats[i]
    if area < 50:
        continue
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / roi.shape[0]
    if 0.2 < ar < 0.8 and 0.2 < hr < 0.5:
        chars.append((x1, y1, w1, h1, area))
        print(f"  pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={hr:.2f}")

print(f"\nTotal: {len(chars)} character-like components")

# Now use segment_characters
print("\n=== Using segment_characters() ===")
seg_result = segment_characters(roi, plate_type='bike', debug=True)
print(f"Segmented: {len(seg_result.boxes)} characters")
print(f"Debug: {seg_result.debug_info}")

# Save binary for inspection
cv2.imwrite(r'F:\CODE\XLA\debug_improved_binary.png', inverted)
print(f"\n✓ Saved improved binary: F:\\CODE\\XLA\\debug_improved_binary.png")

# Draw boxes
debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
for box in seg_result.boxes:
    cv2.rectangle(debug_img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 1)
cv2.imwrite(r'F:\CODE\XLA\debug_improved_segmented.png', debug_img)
print(f"✓ Saved segmented: F:\\CODE\\XLA\\debug_improved_segmented.png")
