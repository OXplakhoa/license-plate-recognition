#!/usr/bin/env python
"""Debug script to understand why candidate plates are being rejected."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.lp_detector import (
    detect_with_character_validation, detect_with_edge_backup,
    detect_multi_preset, validate_plate_roi, compute_character_score,
    detect_character_candidates
)
from src.utils import ensure_grayscale

# Load image
img_path = 'data/test_images/CarTGMT/AQUA7_51443_checkoutex_2020-10-22-9-48a3Jfs9hifV.jpg'
img_bgr = cv2.imread(img_path)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

print("=" * 70)
print("DEBUG: License Plate Detection Issue")
print("=" * 70)
print(f'Image shape: {gray.shape}')
print()

# Step 1: Run detection without character validation (just contour detection)
plates, debug_info = detect_multi_preset(gray, debug=True)
print(f'Step 1 - detect_multi_preset found: {len(plates)} plates')
for i, p in enumerate(plates):
    x, y, w, h = int(p[0]), int(p[1]), int(p[2]), int(p[3])
    ar = w / h if h > 0 else 0
    print(f'  Plate {i+1}: ({x}, {y}, {w}, {h}) - AR={ar:.2f}')
print()

# Step 2: Run edge backup detection
detections, edge_info = detect_with_edge_backup(gray, debug=True)
print(f'Step 2 - detect_with_edge_backup found: {len(detections)} detections')
for i, d in enumerate(detections):
    box = d.get('box', d)
    if isinstance(d, dict):
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        ar = w / h if h > 0 else 0
        print(f'  Detection {i+1}: ({x}, {y}, {w}, {h}) - AR={ar:.2f}, method={d.get("method", "unknown")}')
print()

# Step 3: Manually validate each detection from edge_backup
print("Step 3 - Manual character validation for each detection:")
for i, d in enumerate(detections):
    box = d.get('box')
    if box is None:
        continue
    
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    # Get ROI
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        print(f'  Detection {i+1}: Empty ROI')
        continue
    
    # Validate
    is_valid, score, details = validate_plate_roi(gray, (x, y, w, h))
    
    print(f'  Detection {i+1}: box=({x}, {y}, {w}, {h})')
    print(f'    is_valid: {is_valid}')
    print(f'    char_score: {score:.4f}')
    print(f'    char_count: {details.get("char_count", "N/A")}')
    if 'scores' in details:
        scores = details['scores']
        print(f'    subscores: char_count={scores.get("char_count", 0):.4f}, '
              f'size_consistency={scores.get("size_consistency", 0):.4f}, '
              f'spacing_regularity={scores.get("spacing_regularity", 0):.4f}, '
              f'alignment={scores.get("alignment", 0):.4f}')
    print()

# Step 4: Run full detection with character validation
print("=" * 70)
print("Step 4 - Full detect_with_character_validation:")
valid_dets, char_info = detect_with_character_validation(gray, bgr_image=img_bgr, debug=True)
print(f'  Total candidates: {char_info["candidates_count"]}')
print(f'  Valid count: {char_info["valid_count"]}')
print()

if valid_dets:
    for i, d in enumerate(valid_dets):
        box = d['box']
        char_score = d.get('char_score', 'N/A')
        char_valid = d.get('char_valid', False)
        print(f'  Candidate {i+1}: box={box}, score={char_score}, valid={char_valid}')
else:
    print("  No detections passed validation!")

# Step 5: Check the specific plate region more closely
print()
print("=" * 70)
print("Step 5 - Debug the likely plate region (219, 428, 128, 22):")
plate_box = (219, 428, 128, 22)
x, y, w, h = plate_box
roi = gray[y:y+h, x:x+w]

print(f'  ROI shape: {roi.shape}')

# Detect characters in this ROI
chars, char_info = detect_character_candidates(roi)
print(f'  Characters detected: {len(chars)}')
print(f'  Char info: {char_info}')

# Compute character score
score, details = compute_character_score(roi)
print(f'  Character score: {score:.4f}')
print(f'  Details: {details}')

# Step 6: Deep debug of character detection
print()
print("=" * 70)
print("Step 6 - Deep debug character detection in plate ROI:")

roi_h, roi_w = roi.shape[:2]
roi_area = roi_h * roi_w
print(f'  ROI dimensions: {roi_w}x{roi_h}, area={roi_area}')

# Apply same preprocessing as detect_character_candidates
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Clean morphology
binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, 
                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
inverted = cv2.bitwise_not(binary_clean)

# Find connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
print(f'  Connected components in binary: {num_labels - 1}')  # exclude background

num_labels_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
print(f'  Connected components in inverted: {num_labels_inv - 1}')

# Check filter thresholds used in detect_character_candidates
min_area_ratio = 0.003
max_area_ratio = 0.25
aspect_ratio_range = (0.1, 1.5)
min_height_ratio = 0.15
max_height_ratio = 0.9

min_area = roi_area * min_area_ratio
max_area = roi_area * max_area_ratio
min_height = roi_h * min_height_ratio
max_height = roi_h * max_height_ratio

print(f'\n  Filter thresholds:')
print(f'    min_area: {min_area:.1f} (from {min_area_ratio})')
print(f'    max_area: {max_area:.1f} (from {max_area_ratio})')
print(f'    min_height: {min_height:.1f} (from {min_height_ratio})')
print(f'    max_height: {max_height:.1f} (from {max_height_ratio})')
print(f'    aspect_ratio_range: {aspect_ratio_range}')

# Check each component
print(f'\n  Analyzing binary components:')
for i in range(1, min(num_labels, 20)):  # Skip background (0), limit to first 20
    cx, cy, bw, bh, area = stats[i]
    aspect = bw / float(bh) if bh > 0 else 0
    
    reject_reason = []
    if area < min_area:
        reject_reason.append(f'area too small ({area:.0f} < {min_area:.1f})')
    if area > max_area:
        reject_reason.append(f'area too large ({area:.0f} > {max_area:.1f})')
    if bh < min_height:
        reject_reason.append(f'height too small ({bh} < {min_height:.1f})')
    if bh > max_height:
        reject_reason.append(f'height too large ({bh} > {max_height:.1f})')
    if aspect < aspect_ratio_range[0]:
        reject_reason.append(f'aspect too narrow ({aspect:.2f} < {aspect_ratio_range[0]})')
    if aspect > aspect_ratio_range[1]:
        reject_reason.append(f'aspect too wide ({aspect:.2f} > {aspect_ratio_range[1]})')
    
    margin = min(5, roi_w // 20, roi_h // 20)
    if cx < margin and (cx + bw) > (roi_w - margin):
        reject_reason.append('spans entire width')
    
    status = "REJECTED" if reject_reason else "ACCEPTED"
    print(f'    Component {i}: ({cx}, {cy}, {bw}, {bh}) area={area}, aspect={aspect:.2f} - {status}')
    if reject_reason:
        for r in reject_reason:
            print(f'      - {r}')

print(f'\n  Analyzing INVERTED components:')
for i in range(1, min(num_labels_inv, 20)):  # Skip background (0), limit to first 20
    cx, cy, bw, bh, area = stats_inv[i]
    aspect = bw / float(bh) if bh > 0 else 0
    
    reject_reason = []
    if area < min_area:
        reject_reason.append(f'area too small ({area:.0f} < {min_area:.1f})')
    if area > max_area:
        reject_reason.append(f'area too large ({area:.0f} > {max_area:.1f})')
    if bh < min_height:
        reject_reason.append(f'height too small ({bh} < {min_height:.1f})')
    if bh > max_height:
        reject_reason.append(f'height too large ({bh} > {max_height:.1f})')
    if aspect < aspect_ratio_range[0]:
        reject_reason.append(f'aspect too narrow ({aspect:.2f} < {aspect_ratio_range[0]})')
    if aspect > aspect_ratio_range[1]:
        reject_reason.append(f'aspect too wide ({aspect:.2f} > {aspect_ratio_range[1]})')
    
    margin = min(5, roi_w // 20, roi_h // 20)
    if cx < margin and (cx + bw) > (roi_w - margin):
        reject_reason.append('spans entire width')
    
    status = "REJECTED" if reject_reason else "ACCEPTED"
    print(f'    Component {i}: ({cx}, {cy}, {bw}, {bh}) area={area}, aspect={aspect:.2f} - {status}')
    if reject_reason:
        for r in reject_reason:
            print(f'      - {r}')

# Save debug images
cv2.imwrite('debug_output/debug_roi_gray.png', roi)
cv2.imwrite('debug_output/debug_roi_binary.png', binary_clean)
cv2.imwrite('debug_output/debug_roi_inverted.png', inverted)
print(f'\n  Debug images saved to debug_output/')

# Step 7: Test the full pipeline
print()
print("=" * 70)
print("Step 7 - Testing full pipeline:")
from src.pipeline import LicensePlateRecognizer

recognizer = LicensePlateRecognizer(
    ocr_engine="tesseract",
    use_character_validation=True,
    use_perspective_correction=True,
    use_deskew=True,
    debug=True
)

result = recognizer.recognize(img_bgr)
print(f'  Pipeline result:')
print(f'  - Number of plates: {len(result.plates)}')
if result.best_plate:
    print(f'  - Best plate text: {result.best_plate.text}')
    print(f'  - Confidence: {result.best_plate.confidence:.2f}%')
    print(f'  - Box: {result.best_plate.box}')
    print(f'  - Detection method: {result.best_plate.detection_method}')
else:
    print('  - No plate detected')

# Debug: Check what detections were found
if 'detection' in result.debug_info:
    det_info = result.debug_info['detection']
    print(f'  - Detection info: candidates={det_info.get("candidates_count", "N/A")}, valid={det_info.get("valid_count", "N/A")}')

print(f'  All plates found:')
for i, plate in enumerate(result.plates):
    print(f'    {i+1}: text="{plate.text}", box={plate.box}, conf={plate.confidence:.2f}, method={plate.detection_method}')

# Step 8: Direct OCR test on the plate ROI
print()
print("=" * 70)
print("Step 8 - Direct OCR test on plate ROI (219, 428, 128, 22):")
from src.ocr_engine import configure_tesseract, ocr_plate_line, ocr_plate_multi_psm
from src.heuristics import apply_heuristics, is_valid_plate

configure_tesseract()

plate_box = (219, 428, 128, 22)
x, y, w, h = plate_box
roi = gray[y:y+h, x:x+w]

# OCR with different methods
print(f'  ROI shape: {roi.shape}')

# Method 1: line OCR
line_result = ocr_plate_line(roi, plate_type='car_rect')
print(f'  Line OCR: text="{line_result.text}", conf={line_result.mean_conf:.2f}')

# Method 2: multi PSM OCR
multi_result = ocr_plate_multi_psm(roi, plate_type='car_rect')
print(f'  Multi PSM OCR: text="{multi_result.text}", conf={multi_result.mean_conf:.2f}')

# Check heuristics
text1_heur = apply_heuristics(line_result.text)
text2_heur = apply_heuristics(multi_result.text)
print(f'  After heuristics:')
print(f'    Line: "{text1_heur}" - valid={is_valid_plate(text1_heur)}')
print(f'    Multi: "{text2_heur}" - valid={is_valid_plate(text2_heur)}')

# Try upscaling the ROI
roi_upscaled = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
print(f'  Upscaled ROI shape: {roi_upscaled.shape}')

line_result_up = ocr_plate_line(roi_upscaled, plate_type='car_rect')
print(f'  Upscaled Line OCR: text="{line_result_up.text}", conf={line_result_up.mean_conf:.2f}')

text_up_heur = apply_heuristics(line_result_up.text)
print(f'  After heuristics: "{text_up_heur}" - valid={is_valid_plate(text_up_heur)}')

# Step 9: Test segmentation + OCR (like in pipeline)
print()
print("=" * 70)
print("Step 9 - Test character segmentation on plate ROI:")
from src.character_segmenter import segment_characters_multi_method

# Upscale first
plate_roi = gray[428:428+22, 219:219+128]
scale = 3
plate_roi_up = cv2.resize(plate_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
print(f"  Upscaled plate ROI: {plate_roi_up.shape}")

seg_result = segment_characters_multi_method(plate_roi_up, plate_type='car_rect')
print(f"  Segmentation result:")
print(f"    Binary shape: {seg_result.binary.shape if seg_result.binary is not None else 'None'}")
print(f"    Num chars: {len(seg_result.char_images) if seg_result.char_images else 0}")

if seg_result.char_images:
    from src.ocr_engine import ocr_characters
    ocr_result = ocr_characters(seg_result.char_images, vn_plate=True, plate_type='car_rect')
    print(f"  OCR result: '{ocr_result.text}' (conf={ocr_result.mean_conf:.0f})")
    text_heur = apply_heuristics(ocr_result.text)
    print(f"  After heuristics: '{text_heur}' - valid={is_valid_plate(text_heur)}")

# Step 10: Visualize what the visualization script sees at step 3.5
print()
print("=" * 70)
print("Step 10 - Check contours like visualization script:")

# Same preprocessing as visualization script
gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges_canny = cv2.Canny(gaussian_blur, 50, 150)
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
morph_close = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel_close)
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel_open)

contours, _ = cv2.findContours(morph_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"  Total contours: {len(contours)}")

# Filter like visualization script
h_img, w_img = gray.shape[:2]
plate_candidates = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    ar = w / h if h > 0 else 0
    area = w * h
    if 1.0 < ar < 6.0 and area > 1000:
        plate_candidates.append((x, y, w, h, ar, area))

print(f"  Plate candidates (AR 1-6, area>1000): {len(plate_candidates)}")
for i, (x, y, w, h, ar, area) in enumerate(plate_candidates):
    print(f"    {i+1}: ({x}, {y}, {w}, {h}) AR={ar:.2f}, area={area}")

