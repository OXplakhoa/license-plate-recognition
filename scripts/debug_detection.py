# -*- coding: utf-8 -*-
"""Debug detection to understand why plates not found"""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import numpy as np
from src.lp_detector import detect_multi_preset, detect_license_plates, PRESETS

# Test images
test_images = [
    r'F:\CODE\XLA\data\test_images\CarTGMT\AQUA4_01418_checkin_2020-10-22-13-28x1xGZQit5B.jpg',
    r'F:\CODE\XLA\data\test_images\CarTGMT\AEONTP_5026788_checkin_2020-1-13-16-19_f7zXETONv.jpg',
    r'F:\CODE\XLA\data\test_images\CarTGMT\AQUA2_09566_checkin_2020-10-22-10-45NGiXXm1L2u.jpg',  # This one works
]

for img_path in test_images:
    print("=" * 60)
    print(f"Testing: {img_path.split('\\')[-1][:40]}")
    
    img = cv2.imread(img_path)
    if img is None:
        print("  Failed to load!")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    print(f"  Image size: {w}x{h}, area: {w*h}")
    
    # Test each preset
    for preset in ['car_square', 'car_rect', 'bike']:
        plates, info = detect_license_plates(gray, mode=preset, debug=True)
        cfg = PRESETS[preset]
        
        # Calculate actual thresholds
        min_area = cfg.min_area_ratio * w * h
        max_area = cfg.max_area_ratio * w * h
        
        rejected = info['rejected']
        print(f"\n  {preset}:")
        print(f"    Area range: {min_area:.0f} - {max_area:.0f}")
        print(f"    Aspect range: {cfg.aspect_ratio_range}")
        print(f"    Contours: {info['contours']}")
        print(f"    Rejected - area:{rejected['area']}, aspect:{rejected['aspect']}, solidity:{rejected['solidity']}, margin:{rejected['margin']}")
        print(f"    Plates found: {len(plates)}")
        
        if plates:
            for p in plates:
                x, y, pw, ph = p
                print(f"      Box: ({x},{y},{pw},{ph}) aspect={pw/ph:.2f}")
    
    print()
