"""Check if detection is cropping off left part of plate."""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from src.lp_detector import detect_with_character_validation, correct_plate_perspective_and_skew

# Load image
img = cv2.imread(r'F:\CODE\XLA\data\test_images\CarTGMT\AQUA7_91177_checkin_2020-10-27-9-51IC0IB7Un_5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Original image size: {img.shape}")

# Detect
dets, _ = detect_with_character_validation(gray, min_char_score=0.35, debug=True)
x, y, w, h = dets[0]['box']
print(f"\nDetection box: x={x}, y={y}, w={w}, h={h}")
print(f"Box percentage: x={x/img.shape[1]*100:.1f}%, y={y/img.shape[0]*100:.1f}%")

# Extract ROI with padding to see if plate extends beyond detection
pad = 30
x1 = max(0, x - pad)
y1 = max(0, y - pad)
x2 = min(gray.shape[1], x + w + pad)
y2 = min(gray.shape[0], y + h + pad)

roi_padded = gray[y1:y2, x1:x2]
roi_original = gray[y:y+h, x:x+w]

# Save for visual inspection
cv2.imwrite(r'F:\CODE\XLA\debug_roi_original.png', roi_original)
cv2.imwrite(r'F:\CODE\XLA\debug_roi_padded.png', roi_padded)

# Mark detection on original
img_marked = img.copy()
cv2.rectangle(img_marked, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Mark padded area
cv2.rectangle(img_marked, (x1, y1), (x2, y2), (255, 0, 0), 1)
cv2.imwrite(r'F:\CODE\XLA\debug_detection_marked.png', img_marked)

print(f"\n✓ Saved original ROI: F:\\CODE\\XLA\\debug_roi_original.png")
print(f"✓ Saved padded ROI: F:\\CODE\\XLA\\debug_roi_padded.png")  
print(f"✓ Saved detection marked: F:\\CODE\\XLA\\debug_detection_marked.png")

# Check what's on the left of the detection
print(f"\n=== Checking area LEFT of detection ===")
if x > 10:
    left_region = gray[y:y+h, max(0,x-40):x]
    print(f"Left region size: {left_region.shape}")
    print(f"Left region mean intensity: {left_region.mean():.1f}")
    # If there are darker pixels (text), mean should be lower
    
# Also check the filename for ground truth
print(f"\n=== Filename analysis ===")
filename = 'AQUA7_91177_checkin_2020-10-27-9-51IC0IB7Un_5.jpg'
# The filename contains "91177" which suggests plate ends with 91177
# So full plate might be "30E-91177" or similar
print(f"Filename: {filename}")
print(f"Contains '91177' - this is likely the last 5 digits")

# Check ground truth
print(f"\n=== Ground truth ===")
with open(r'F:\CODE\XLA\data\ground_truth.txt', 'r') as f:
    for line in f:
        if 'AQUA7_91177' in line:
            print(f"Found: {line.strip()}")
            break
