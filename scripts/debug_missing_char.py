"""Debug why character at (89,11) is not found."""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np

# Load ROI
roi = cv2.imread(r'F:\CODE\XLA\debug_no_perspective_roi.png', cv2.IMREAD_GRAYSCALE)
print(f"ROI size: {roi.shape}")

# Try pure adaptive (which found 8 chars earlier)
adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

# Get all components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(adaptive, connectivity=8)

print("\n=== All components in TOP row (y < 40) with pure adaptive ===")
h, w = roi.shape
area = h * w
min_area = area * 0.002  # 0.2%
max_area = area * 0.20   # 20%

for i in range(1, num_labels):
    x1, y1, w1, h1, comp_area = stats[i]
    if y1 >= 40:  # Only top row
        continue
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / h
    area_pct = comp_area / area * 100
    
    print(f"  #{i}: pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={hr:.2f} Area={comp_area} ({area_pct:.2f}%)")
    
    # Check which filter would reject this
    reasons = []
    if comp_area < min_area:
        reasons.append(f"area too small ({comp_area} < {min_area:.0f})")
    if comp_area > max_area:
        reasons.append(f"area too big ({comp_area} > {max_area:.0f})")
    if not (0.10 <= ar <= 1.5):
        reasons.append(f"AR out of range ({ar:.2f} not in [0.10, 1.5])")
    if hr < 0.15:
        reasons.append(f"height too small ({hr:.2f} < 0.15)")
        
    if reasons:
        print(f"      → Would be REJECTED: {', '.join(reasons)}")
    else:
        print(f"      → Would PASS filters ✓")

# Now try with light preprocessing  
print("\n=== Light preprocessing (just blur) ===")
blurred = cv2.GaussianBlur(roi, (3, 3), 0)
# Try adaptive on blurred
adaptive_light = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

num_labels2, _, stats2, _ = cv2.connectedComponentsWithStats(adaptive_light, connectivity=8)

print(f"Components in top row (y < 40):")
for i in range(1, num_labels2):
    x1, y1, w1, h1, comp_area = stats2[i]
    if y1 >= 40:
        continue
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / h
    
    if comp_area > 50 and 0.10 <= ar <= 1.5 and hr >= 0.15:
        print(f"  #{i}: pos=({x1:3d},{y1:2d}) size={w1:2d}x{h1:2d} AR={ar:.2f} H%={hr:.2f} Area={comp_area}")

# The key insight: which character is at x=89?
# From earlier: pos=(89,11) size=15x32 AR=0.47
# This should be found!

print("\n=== Detailed check for component near x=89 ===")
for i in range(1, num_labels2):
    x1, y1, w1, h1, comp_area = stats2[i]
    if 85 <= x1 <= 95 and y1 < 40:
        ar = w1 / h1 if h1 > 0 else 0
        hr = h1 / h
        print(f"  Found at ({x1},{y1}): {w1}x{h1}, AR={ar:.2f}, H%={hr:.2f}, Area={comp_area}")

cv2.imwrite(r'F:\CODE\XLA\debug_pure_adaptive_inv.png', adaptive)
cv2.imwrite(r'F:\CODE\XLA\debug_light_adaptive_inv.png', adaptive_light)
