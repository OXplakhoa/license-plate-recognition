# -*- coding: utf-8 -*-
"""Test detection rate on full dataset"""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
from pathlib import Path
from src.lp_detector import detect_multi_preset

test_dir = Path(r"F:\CODE\XLA\data\test_images\CarTGMT")
images = list(test_dir.glob("*.jpg"))

print(f"Testing {len(images)} images...")

detected = 0
failed = 0
multi_detected = 0

failed_list = []

for img_path in images[:200]:  # Test first 200
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates, _ = detect_multi_preset(gray)
    
    if len(plates) == 0:
        failed += 1
        failed_list.append(img_path.name)
    elif len(plates) == 1:
        detected += 1
    else:
        multi_detected += 1
        detected += 1

total = detected + failed
print(f"\nResults (first 100 images):")
print(f"  Detected: {detected}/{total} ({100*detected/total:.1f}%)")
print(f"  Multi-detection: {multi_detected}")
print(f"  Failed: {failed}/{total} ({100*failed/total:.1f}%)")
print(f"\nFailed samples (first 10):")
for f in failed_list[:10]:
    print(f"  - {f}")
