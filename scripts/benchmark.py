# -*- coding: utf-8 -*-
"""Comprehensive benchmark for license plate recognition"""
import sys
sys.path.insert(0, r'F:\CODE\XLA')
import cv2
import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.pipeline import LicensePlateRecognizer


def extract_expected_plate_from_filename(filename: str) -> str:
    """Try to extract expected plate number from filename.
    
    Patterns like AQUA7_51624_... or AQUA4_26A01418_...
    """
    # Remove AQUA/AEON prefix and common suffixes
    name = filename.replace('.jpg', '').replace('.png', '')
    
    # Look for pattern after first underscore
    parts = name.split('_')
    if len(parts) >= 2:
        candidate = parts[1]
        # Filter likely plate patterns
        # VN plates: 2 digits + letter + 5-6 digits (e.g., 51F86947, 26A01418)
        # Or just 5 digits (partial)
        if re.match(r'^[0-9]{2}[A-Z][0-9]{4,6}$', candidate):
            return candidate
        elif re.match(r'^[0-9]{5,6}$', candidate):
            return candidate  # Partial plate number
    
    return ""


def normalize_plate(text: str) -> str:
    """Normalize plate text for comparison."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def calculate_accuracy(predicted: str, expected: str) -> float:
    """Calculate character-level accuracy."""
    if not expected:
        return 0.0
    
    pred = normalize_plate(predicted)
    exp = normalize_plate(expected)
    
    if not exp:
        return 0.0
    
    # Levenshtein-like: count matching chars
    matches = 0
    min_len = min(len(pred), len(exp))
    
    for i in range(min_len):
        if pred[i] == exp[i]:
            matches += 1
    
    return matches / len(exp) if exp else 0.0


def run_benchmark(num_images: int = 100, use_perspective_correction: bool = True) -> Dict:
    """Run benchmark on test images."""
    test_dir = Path(r"F:\CODE\XLA\data\test_images\CarTGMT")
    images = list(test_dir.glob("*.jpg"))[:num_images]
    
    recognizer = LicensePlateRecognizer(
        use_character_validation=True,
        use_perspective_correction=use_perspective_correction,
        min_char_score=0.35,
        ocr_method="segment",
    )
    
    results = {
        'total': 0,
        'detected': 0,
        'matched': 0,
        'confidences': [],
        'accuracies': [],
        'failed': [],
        'samples': [],
    }
    
    for img_path in images:
        results['total'] += 1
        
        expected = extract_expected_plate_from_filename(img_path.name)
        
        # Recognize
        result = recognizer.recognize_file(str(img_path), max_plates=1)
        
        if result.best_plate:
            results['detected'] += 1
            predicted = result.best_plate.text
            conf = result.best_plate.confidence
            
            results['confidences'].append(conf)
            
            if expected:
                accuracy = calculate_accuracy(predicted, expected)
                results['accuracies'].append(accuracy)
                
                if accuracy >= 0.8:  # 80% char match
                    results['matched'] += 1
            
            # Save sample
            if len(results['samples']) < 10:
                results['samples'].append({
                    'file': img_path.name,
                    'expected': expected,
                    'predicted': predicted,
                    'confidence': conf,
                    'accuracy': accuracy if expected else None,
                })
        else:
            results['failed'].append(img_path.name)
    
    return results


def main():
    print("Running benchmark on 100 images...")
    print()
    
    results = run_benchmark(100)
    
    # Summary
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total images: {results['total']}")
    print(f"Detection rate: {results['detected']}/{results['total']} ({100*results['detected']/results['total']:.1f}%)")
    print(f"Match rate (≥80% accuracy): {results['matched']}/{results['detected']} ({100*results['matched']/results['detected']:.1f}%)" if results['detected'] else "N/A")
    
    if results['confidences']:
        avg_conf = sum(results['confidences']) / len(results['confidences'])
        print(f"Average OCR confidence: {avg_conf:.1f}%")
    
    if results['accuracies']:
        avg_acc = sum(results['accuracies']) / len(results['accuracies'])
        print(f"Average character accuracy: {avg_acc*100:.1f}%")
    
    print()
    print("Sample predictions:")
    for s in results['samples']:
        acc_str = f", acc={s['accuracy']*100:.0f}%" if s.get('accuracy') is not None else ""
        print(f"  {s['file'][:35]}:")
        print(f"    Expected: {s['expected'] or 'N/A'}")
        print(f"    Predicted: '{s['predicted']}' (conf={s['confidence']:.0f}%{acc_str})")
    
    print()
    print(f"Failed detections: {len(results['failed'])} images")


if __name__ == "__main__":
    main()
