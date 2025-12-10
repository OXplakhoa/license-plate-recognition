# -*- coding: utf-8 -*-
"""Debug character validation scores"""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
from src.lp_detector import detect_multi_preset, compute_character_score

# Test on sample image - one with good detection
img = cv2.imread(r'F:\CODE\XLA\data\test_images\CarTGMT\AEONTP_51F86947_checkin_2020-1-13-16-15sUVxP1Ihlt.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get all detections
detections, _ = detect_multi_preset(gray)

print(f'Total detections: {len(detections)}')
print()

for i, (x, y, w, h) in enumerate(detections):
    roi = gray[y:y+h, x:x+w]
    score, details = compute_character_score(roi)
    aspect = w / h if h > 0 else 0
    
    print(f'Detection {i+1}: box=({x},{y},{w},{h}), aspect={aspect:.2f}')
    print(f'  Score: {score:.2f}')
    print(f'  Chars: {details.get("char_count", 0)}')
    scores = details.get('scores', {})
    if scores:
        print(f'  - count: {scores.get("char_count", 0):.2f}')
        print(f'  - size: {scores.get("size_consistency", 0):.2f}')
        print(f'  - spacing: {scores.get("spacing_regularity", 0):.2f}')
        print(f'  - align: {scores.get("alignment", 0):.2f}')
    print()
