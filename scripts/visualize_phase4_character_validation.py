"""
Phase 4: Character-based ROI Validation - Visualization Script
===============================================================

This script demonstrates the character-based ROI validation for license plate
detection. It shows how character detection can filter false positive detections.

Usage:
    python scripts/visualize_phase4_character_validation.py

Output:
    - Console output with validation results
    - Visualization images saved to data/test_images/
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lp_detector import (
    detect_multi_preset,
    detect_with_edge_backup,
    detect_character_candidates,
    compute_character_score,
    validate_plate_roi,
    filter_detections_by_characters,
    detect_with_character_validation,
)


def visualize_character_detection(image_path: str, save_path: str = None):
    """Visualize character detection in detected plates."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_name = os.path.basename(image_path)
    
    # Detect plates
    detections, detect_info = detect_with_edge_backup(gray, debug=True)
    
    if not detections:
        print(f"  {img_name}: No plates detected")
        return
    
    # Create visualization
    num_plates = min(3, len(detections))
    fig, axes = plt.subplots(num_plates, 4, figsize=(16, 4 * num_plates))
    if num_plates == 1:
        axes = axes.reshape(1, -1)
    
    for idx, det in enumerate(detections[:num_plates]):
        x, y, w, h = det['box']
        
        # Extract ROI
        roi = gray[y:y+h, x:x+w]
        roi_bgr = img[y:y+h, x:x+w]
        
        # Get character candidates
        chars, char_info = detect_character_candidates(roi)
        
        # Compute score
        score, score_details = compute_character_score(roi)
        
        # Column 1: Original ROI
        axes[idx, 0].imshow(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f'ROI {idx+1}: {w}x{h}\nMethod: {det.get("method", "unknown")}')
        axes[idx, 0].axis('off')
        
        # Column 2: Binarized with characters marked
        # Binarize for visualization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
        _, binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        for (cx, cy, cw, ch) in chars:
            cv2.rectangle(binary_vis, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 1)
        
        axes[idx, 1].imshow(binary_vis)
        axes[idx, 1].set_title(f'Binary + Chars\n{len(chars)} candidates')
        axes[idx, 1].axis('off')
        
        # Column 3: ROI with characters marked
        roi_vis = roi_bgr.copy()
        for (cx, cy, cw, ch) in chars:
            cv2.rectangle(roi_vis, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)
        
        axes[idx, 2].imshow(cv2.cvtColor(roi_vis, cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title(f'Detected Chars: {len(chars)}')
        axes[idx, 2].axis('off')
        
        # Column 4: Score details
        scores = score_details.get('scores', {})
        score_text = f"Total Score: {score:.3f}\n\n"
        score_text += f"Char Count: {scores.get('char_count', 0):.3f}\n"
        score_text += f"Size Consistency: {scores.get('size_consistency', 0):.3f}\n"
        score_text += f"Spacing: {scores.get('spacing_regularity', 0):.3f}\n"
        score_text += f"Alignment: {scores.get('alignment', 0):.3f}\n"
        score_text += f"\nValid: {'✓' if score >= 0.35 else '✗'}"
        
        axes[idx, 3].text(0.1, 0.5, score_text, fontsize=11, 
                         transform=axes[idx, 3].transAxes, verticalalignment='center',
                         family='monospace')
        axes[idx, 3].axis('off')
        axes[idx, 3].set_title('Validation Scores')
    
    plt.suptitle(f'{img_name[:40]}...', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()
    plt.close()


def compare_with_without_validation(test_folder: str, num_images: int = 8, save_folder: str = None):
    """Compare detection results with and without character validation."""
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    images = images[:num_images]
    
    print(f"\n{'='*60}")
    print("PHASE 4: CHARACTER-BASED ROI VALIDATION")
    print(f"{'='*60}")
    print(f"Testing on {len(images)} images from {test_folder}\n")
    
    results_summary = []
    
    # Create figure for comparison
    fig, axes = plt.subplots(len(images), 3, figsize=(15, 4 * len(images)))
    if len(images) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_name in enumerate(images):
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Cannot load: {img_name}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        short_name = img_name[:25] + "..." if len(img_name) > 25 else img_name
        
        # Method 1: Without validation (edge backup)
        raw_detections, _ = detect_with_edge_backup(gray, debug=True)
        
        # Method 2: With character validation
        valid_detections, valid_info = detect_with_character_validation(gray, debug=True)
        
        # Count valid vs invalid
        valid_count = valid_info['valid_count']
        invalid_count = valid_info['char_validation']['invalid']
        
        # Visualize
        # Column 1: Original
        axes[idx, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f'{short_name}', fontsize=9)
        axes[idx, 0].axis('off')
        
        # Column 2: All detections (without validation)
        raw_img = img.copy()
        for det in raw_detections[:5]:
            x, y, w, h = det['box']
            cv2.rectangle(raw_img, (x, y), (x+w, y+h), (255, 165, 0), 2)
        axes[idx, 1].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f'Without Validation: {len(raw_detections)}', fontsize=9)
        axes[idx, 1].axis('off')
        
        # Column 3: Valid detections only
        valid_img = img.copy()
        for det in valid_detections:
            if det.get('char_valid', False):
                x, y, w, h = det['box']
                score = det.get('char_score', 0)
                color = (0, 255, 0) if score >= 0.5 else (0, 200, 255)
                cv2.rectangle(valid_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(valid_img, f'{score:.2f}', (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        axes[idx, 2].imshow(cv2.cvtColor(valid_img, cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title(f'With Validation: {valid_count} (filtered: {invalid_count})', fontsize=9)
        axes[idx, 2].axis('off')
        
        # Summary
        results_summary.append({
            'name': short_name,
            'raw': len(raw_detections),
            'valid': valid_count,
            'filtered': invalid_count
        })
        
        print(f"  {short_name}: Raw={len(raw_detections)}, Valid={valid_count}, Filtered={invalid_count}")
    
    plt.tight_layout()
    
    # Save figure
    if save_folder:
        save_path = os.path.join(save_folder, 'phase4_character_validation_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved comparison to: {save_path}")
    
    plt.show()
    plt.close()
    
    # Print summary table
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Image':<30} {'Raw':>8} {'Valid':>8} {'Filtered':>10}")
    print("-" * 60)
    
    total_raw = 0
    total_valid = 0
    total_filtered = 0
    
    for r in results_summary:
        print(f"{r['name']:<30} {r['raw']:>8} {r['valid']:>8} {r['filtered']:>10}")
        total_raw += r['raw']
        total_valid += r['valid']
        total_filtered += r['filtered']
    
    print("-" * 60)
    print(f"{'TOTAL':<30} {total_raw:>8} {total_valid:>8} {total_filtered:>10}")
    
    return results_summary


def analyze_character_scores(test_folder: str, num_images: int = 10):
    """Analyze character validation scores for detected regions."""
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])
    
    print(f"\n{'='*60}")
    print("CHARACTER SCORE ANALYSIS")
    print(f"{'='*60}\n")
    
    all_scores = []
    valid_scores = []
    invalid_scores = []
    
    for img_name in images[:num_images]:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get all detections with validation info
        detections, info = detect_with_character_validation(gray, debug=True)
        
        for det in detections:
            score = det.get('char_score', 0)
            is_valid = det.get('char_valid', False)
            char_count = det.get('char_count', 0)
            
            all_scores.append(score)
            if is_valid:
                valid_scores.append(score)
            else:
                invalid_scores.append(score)
    
    if all_scores:
        print(f"Total detections analyzed: {len(all_scores)}")
        print(f"Valid: {len(valid_scores)}, Invalid: {len(invalid_scores)}")
        print(f"\nScore Statistics:")
        print(f"  All - Mean: {np.mean(all_scores):.3f}, Std: {np.std(all_scores):.3f}")
        if valid_scores:
            print(f"  Valid - Mean: {np.mean(valid_scores):.3f}, Std: {np.std(valid_scores):.3f}")
        if invalid_scores:
            print(f"  Invalid - Mean: {np.mean(invalid_scores):.3f}, Std: {np.std(invalid_scores):.3f}")
    
    return all_scores, valid_scores, invalid_scores


def main():
    """Main function to run visualizations."""
    
    # Test folder path
    test_folder = r'F:\CODE\XLA\data\test_images\CarTGMT'
    save_folder = r'F:\CODE\XLA\data\test_images'
    
    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        return
    
    # 1. Compare with/without validation
    print("\n" + "=" * 60)
    print("1. COMPARING WITH/WITHOUT CHARACTER VALIDATION")
    print("=" * 60)
    compare_with_without_validation(test_folder, num_images=8, save_folder=save_folder)
    
    # 2. Analyze character scores
    print("\n" + "=" * 60)
    print("2. CHARACTER SCORE ANALYSIS")
    print("=" * 60)
    analyze_character_scores(test_folder, num_images=15)
    
    # 3. Detailed visualization for one image
    print("\n" + "=" * 60)
    print("3. DETAILED CHARACTER DETECTION VISUALIZATION")
    print("=" * 60)
    
    # Find an image with detected plates
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])
    for img_name in images[:20]:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections, _ = detect_with_edge_backup(gray)
        if detections:
            print(f"\nShowing detailed visualization for: {img_name}")
            save_path = os.path.join(save_folder, 'phase4_character_detection_detail.png')
            visualize_character_detection(img_path, save_path=save_path)
            break
    
    print("\n" + "=" * 60)
    print("✅ PHASE 4 VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
