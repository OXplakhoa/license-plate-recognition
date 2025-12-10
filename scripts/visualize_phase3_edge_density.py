"""
Phase 3: Edge Density Backup Detector - Visualization Script
=============================================================

This script demonstrates the edge density backup detector for license plate
detection. It compares contour-based detection with the edge density fallback
and visualizes the results.

Usage:
    python scripts/visualize_phase3_edge_density.py

Output:
    - Console output with detection results
    - Visualization images saved to data/test_images/
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lp_detector import (
    detect_multi_preset,
    sliding_window_detect,
    detect_with_edge_backup,
    compute_edge_density,
    compute_vertical_edge_score,
    compute_contrast_score,
    compute_plate_score,
)


def visualize_edge_metrics(image_path: str, save_path: str = None):
    """Visualize edge density metrics for an image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Compute edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Compute Sobel for directional edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normalize for visualization
    sobel_x_vis = cv2.normalize(np.abs(sobel_x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_y_vis = cv2.normalize(np.abs(sobel_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original, Edges, Combined
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original ({w}x{h})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(edges, cmap='gray')
    edge_density = np.count_nonzero(edges) / edges.size
    axes[0, 2].set_title(f'Canny Edges\nDensity: {edge_density:.3f}')
    axes[0, 2].axis('off')
    
    # Row 2: Sobel X, Sobel Y, Detection result
    axes[1, 0].imshow(sobel_x_vis, cmap='gray')
    axes[1, 0].set_title('Vertical Edges (Sobel X)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sobel_y_vis, cmap='gray')
    axes[1, 1].set_title('Horizontal Edges (Sobel Y)')
    axes[1, 1].axis('off')
    
    # Run detection and show result
    detections, debug_info = detect_with_edge_backup(gray, debug=True)
    
    result_img = img.copy()
    for det in detections:
        x, y, bw, bh = det['box']
        color = (0, 255, 0) if det['method'] == 'contour' else (255, 165, 0)
        cv2.rectangle(result_img, (x, y), (x+bw, y+bh), color, 2)
        label = f"{det['method'][:4]} {det['score']:.2f}"
        cv2.putText(result_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    axes[1, 2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Detection Result\nBackup: {debug_info["backup_used"]}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()
    plt.close()


def compare_detection_methods(test_folder: str, num_images: int = 10, save_folder: str = None):
    """Compare contour-based vs edge density detection on multiple images."""
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    images = images[:num_images]
    
    print(f"\n{'='*60}")
    print("PHASE 3: EDGE DENSITY BACKUP DETECTOR")
    print(f"{'='*60}")
    print(f"Testing on {len(images)} images from {test_folder}\n")
    
    results_summary = []
    
    # Create figure for comparison
    fig, axes = plt.subplots(len(images), 4, figsize=(20, 5 * len(images)))
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
        
        # Method 1: Contour-based only
        contour_plates, contour_info = detect_multi_preset(gray, debug=True)
        
        # Method 2: Edge density sliding window
        edge_plates, edge_info = sliding_window_detect(gray, score_threshold=0.45, max_candidates=5, debug=True)
        
        # Method 3: Combined with fallback
        combined_plates, combined_info = detect_with_edge_backup(gray, debug=True)
        
        # Visualize
        # Column 1: Original
        axes[idx, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f'{short_name}', fontsize=9)
        axes[idx, 0].axis('off')
        
        # Column 2: Contour detection
        contour_img = img.copy()
        for (x, y, w, h) in contour_plates:
            cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        axes[idx, 1].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f'Contour: {len(contour_plates)} plates', fontsize=9)
        axes[idx, 1].axis('off')
        
        # Column 3: Edge density detection
        edge_img = img.copy()
        for det in edge_plates[:5]:  # Top 5
            x, y, w, h = det['box']
            cv2.rectangle(edge_img, (x, y), (x+w, y+h), (255, 165, 0), 2)
            score_text = f"{det['score']:.2f}"
            cv2.putText(edge_img, score_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        axes[idx, 2].imshow(cv2.cvtColor(edge_img, cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title(f'Edge Density: {len(edge_plates)} candidates', fontsize=9)
        axes[idx, 2].axis('off')
        
        # Column 4: Combined result
        combined_img = img.copy()
        for det in combined_plates:
            x, y, w, h = det['box']
            color = (0, 255, 0) if det['method'] == 'contour' else (255, 165, 0)
            cv2.rectangle(combined_img, (x, y), (x+w, y+h), color, 2)
        backup_text = "✓" if combined_info['backup_used'] else "✗"
        axes[idx, 3].imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        axes[idx, 3].set_title(f'Combined: {len(combined_plates)} (Backup: {backup_text})', fontsize=9)
        axes[idx, 3].axis('off')
        
        # Summary
        results_summary.append({
            'name': short_name,
            'contour': len(contour_plates),
            'edge': len(edge_plates),
            'combined': len(combined_plates),
            'backup': combined_info['backup_used']
        })
        
        print(f"  {short_name}:")
        print(f"    Contour: {len(contour_plates)}, Edge: {len(edge_plates)}, Combined: {len(combined_plates)}, Backup: {combined_info['backup_used']}")
    
    plt.tight_layout()
    
    # Save figure
    if save_folder:
        save_path = os.path.join(save_folder, 'phase3_edge_density_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved comparison to: {save_path}")
    
    plt.show()
    plt.close()
    
    # Print summary table
    print(f"\n{'='*60}")
    print("DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Image':<30} {'Contour':>8} {'Edge':>8} {'Combined':>10} {'Backup':>8}")
    print("-" * 60)
    
    contour_total = 0
    edge_total = 0
    combined_total = 0
    backup_count = 0
    
    for r in results_summary:
        print(f"{r['name']:<30} {r['contour']:>8} {r['edge']:>8} {r['combined']:>10} {str(r['backup']):>8}")
        contour_total += r['contour']
        edge_total += r['edge']
        combined_total += r['combined']
        if r['backup']:
            backup_count += 1
    
    print("-" * 60)
    print(f"{'TOTAL':<30} {contour_total:>8} {edge_total:>8} {combined_total:>10} {backup_count:>8}")
    
    return results_summary


def analyze_plate_scores(test_folder: str, num_images: int = 5):
    """Analyze plate scores for detected regions."""
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n{'='*60}")
    print("PLATE SCORE ANALYSIS")
    print(f"{'='*60}\n")
    
    for img_name in images[:num_images]:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get detections
        plates, _ = detect_multi_preset(gray)
        
        if not plates:
            print(f"{img_name[:30]}: No plates detected")
            continue
        
        print(f"{img_name[:30]}:")
        
        for i, (x, y, w, h) in enumerate(plates[:3]):
            roi = gray[y:y+h, x:x+w]
            total_score, scores = compute_plate_score(roi)
            
            print(f"  Plate {i+1} ({w}x{h}):")
            print(f"    Edge Density:    {scores['edge_density']:.3f}")
            print(f"    Vertical Edges:  {scores['vertical_edges']:.3f}")
            print(f"    Contrast:        {scores['contrast']:.3f}")
            print(f"    TOTAL SCORE:     {total_score:.3f}")
        print()


def main():
    """Main function to run visualizations."""
    
    # Test folder path
    test_folder = r'F:\CODE\XLA\data\test_images\CarTGMT'
    save_folder = r'F:\CODE\XLA\data\test_images'
    
    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        return
    
    # 1. Compare detection methods
    print("\n" + "=" * 60)
    print("1. COMPARING DETECTION METHODS")
    print("=" * 60)
    compare_detection_methods(test_folder, num_images=8, save_folder=save_folder)
    
    # 2. Analyze plate scores
    print("\n" + "=" * 60)
    print("2. PLATE SCORE ANALYSIS")
    print("=" * 60)
    analyze_plate_scores(test_folder, num_images=5)
    
    # 3. Detailed visualization for one image
    print("\n" + "=" * 60)
    print("3. DETAILED EDGE METRICS VISUALIZATION")
    print("=" * 60)
    
    # Find an image with detected plates
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])
    for img_name in images[:20]:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates, _ = detect_multi_preset(gray)
        if plates:
            print(f"\nShowing detailed visualization for: {img_name}")
            save_path = os.path.join(save_folder, 'phase3_edge_metrics_detail.png')
            visualize_edge_metrics(img_path, save_path=save_path)
            break
    
    print("\n" + "=" * 60)
    print("✅ PHASE 3 VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
