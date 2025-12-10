# -*- coding: utf-8 -*-
"""Analyze contours to understand why detection fails"""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from src.lp_detector import auto_canny, PRESETS

# Test an image that fails
img_path = r'F:\CODE\XLA\data\test_images\CarTGMT\AQUA4_01418_checkin_2020-10-22-13-28x1xGZQit5B.jpg'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
img_area = w * h

print(f"Image: {w}x{h}, area: {img_area}")

cfg = PRESETS['car_square']

# Run edge detection
lower, upper = auto_canny(gray, sigma=cfg.sigma)
edges = cv2.Canny(gray, lower, upper)
print(f"Canny thresholds: {lower}, {upper}")

# Morphology
close_size = cfg.kernel_close or (25, 5)
open_size = cfg.kernel_open or (5, 5)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, close_size))
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, open_size))

# Find contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\nTotal contours: {len(contours)}")
print(f"min_area threshold: {cfg.min_area_ratio * img_area:.0f}")
print(f"max_area threshold: {cfg.max_area_ratio * img_area:.0f}")
print(f"aspect_range: {cfg.aspect_ratio_range}")
print()

# Analyze all contours
print("All contours (sorted by area):")
contour_data = []
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect = bw / (bh or 1)
    area_ratio = area / img_area
    contour_data.append({
        'idx': i,
        'area': area,
        'area_ratio': area_ratio,
        'box': (x, y, bw, bh),
        'aspect': aspect,
    })

# Sort by area descending
contour_data.sort(key=lambda x: x['area'], reverse=True)

for i, c in enumerate(contour_data[:15]):  # Top 15
    min_ok = c['area'] >= cfg.min_area_ratio * img_area
    max_ok = c['area'] <= cfg.max_area_ratio * img_area
    asp_ok = cfg.aspect_ratio_range[0] <= c['aspect'] <= cfg.aspect_ratio_range[1]
    
    status = []
    if not min_ok: status.append("TOO_SMALL")
    if not max_ok: status.append("TOO_LARGE")
    if not asp_ok: status.append(f"ASPECT({c['aspect']:.2f})")
    
    status_str = ", ".join(status) if status else "OK"
    
    print(f"  #{i+1}: area={c['area']:.0f} ({c['area_ratio']*100:.2f}%), "
          f"box={c['box']}, aspect={c['aspect']:.2f} -> {status_str}")

# Save visualization
vis = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
for c in contour_data[:5]:  # Top 5
    x, y, bw, bh = c['box']
    cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
    cv2.putText(vis, f"a={c['area']:.0f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imwrite(r'F:\CODE\XLA\debug_output\contours_analysis.jpg', vis)
print(f"\nSaved visualization to debug_output/contours_analysis.jpg")
