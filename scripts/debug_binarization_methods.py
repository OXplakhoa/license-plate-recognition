"""Analyze why some characters are merged with background."""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np

# Load the original ROI
roi = cv2.imread(r'F:\CODE\XLA\debug_no_perspective_roi.png', cv2.IMREAD_GRAYSCALE)
print(f"ROI size: {roi.shape}")

# Try different binarization approaches to separate characters

print("\n=== Method 1: Adaptive Threshold with different params ===")
# More aggressive block size for small characters  
adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(adaptive, connectivity=8)
print(f"Adaptive (15,5): {num_labels-1} components")

# Check characters
chars = []
for i in range(1, num_labels):
    x1, y1, w1, h1, area = stats[i]
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / roi.shape[0]
    if 50 < area < 1000 and 0.2 < ar < 0.8 and 0.2 < hr < 0.5:
        chars.append((x1, y1, w1, h1, area))
        print(f"  Char at ({x1},{y1}): {w1}x{h1}, AR={ar:.2f}")
print(f"Total chars: {len(chars)}")
cv2.imwrite(r'F:\CODE\XLA\debug_adaptive.png', adaptive)

print("\n=== Method 2: Adaptive with smaller block ===")
adaptive2 = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(adaptive2, connectivity=8)
chars2 = []
for i in range(1, num_labels):
    x1, y1, w1, h1, area = stats[i]
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / roi.shape[0]
    if 50 < area < 1000 and 0.2 < ar < 0.8 and 0.2 < hr < 0.5:
        chars2.append((x1, y1, w1, h1, area))
        print(f"  Char at ({x1},{y1}): {w1}x{h1}, AR={ar:.2f}")
print(f"Total chars: {len(chars2)}")
cv2.imwrite(r'F:\CODE\XLA\debug_adaptive2.png', adaptive2)

print("\n=== Method 3: CLAHE + Otsu ===")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
enhanced = clahe.apply(roi)
_, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu, connectivity=8)
chars3 = []
for i in range(1, num_labels):
    x1, y1, w1, h1, area = stats[i]
    ar = w1 / h1 if h1 > 0 else 0
    hr = h1 / roi.shape[0]
    if 50 < area < 1000 and 0.2 < ar < 0.8 and 0.2 < hr < 0.5:
        chars3.append((x1, y1, w1, h1, area))
        print(f"  Char at ({x1},{y1}): {w1}x{h1}, AR={ar:.2f}")
print(f"Total chars: {len(chars3)}")
cv2.imwrite(r'F:\CODE\XLA\debug_clahe_otsu.png', otsu)

print("\n=== Method 4: Edge-based segmentation ===")
# Use edges to find character boundaries
blurred = cv2.GaussianBlur(roi, (3, 3), 0)
edges = cv2.Canny(blurred, 50, 150)
# Dilate to connect edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated = cv2.dilate(edges, kernel, iterations=1)
cv2.imwrite(r'F:\CODE\XLA\debug_edges.png', dilated)

print("\n=== Method 5: Analyze grayscale intensity profile ===")
# Look at intensity in the top row region (where "30E" should be)
top_region = roi[5:40, :]  # Top 40 pixels where top line should be
print(f"Top region mean: {top_region.mean():.1f}")
# Plot column-wise intensity
col_mean = top_region.mean(axis=0)
print(f"Column intensity range: {col_mean.min():.1f} - {col_mean.max():.1f}")
# Find dark columns (potential character columns)
dark_cols = np.where(col_mean < col_mean.mean())[0]
print(f"Dark columns (below mean): {dark_cols[:30]}...")

# Check specific regions where "3", "0", "E" should be
# If EasyOCR found "30E", we need to find these in the binary image
print("\n=== Checking expected character locations ===")
# Expected x positions for top row characters based on similar bike plates
# Characters are usually in the right 2/3 of the plate
expected_top_x_start = 70  # Starting x for "3"
expected_top_x_end = 150   # Ending x for "E"

print(f"Checking region x=[{expected_top_x_start}:{expected_top_x_end}], y=[5:40]")
check_region = roi[5:40, expected_top_x_start:expected_top_x_end]
print(f"Region mean: {check_region.mean():.1f}")
print(f"Region min: {check_region.min()}, max: {check_region.max()}")

# Threshold just this region
_, region_bin = cv2.threshold(check_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite(r'F:\CODE\XLA\debug_top_region.png', region_bin)
print(f"✓ Saved top region binary")

# Count components in this region
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region_bin, connectivity=8)
print(f"Components in top region: {num_labels - 1}")
for i in range(1, min(num_labels, 10)):
    x1, y1, w1, h1, area = stats[i]
    if area > 30:
        print(f"  Component {i}: pos=({x1},{y1}), size={w1}x{h1}, area={area}")
