"""
End-to-End Pipeline Visualization Script
=========================================

This script demonstrates the complete license plate recognition pipeline
from image input to OCR output with visualizations at each stage.

Usage:
    python scripts/visualize_pipeline.py
    python scripts/visualize_pipeline.py --image <path_to_image>

Output:
    - Console output with recognition results
    - Visualization images saved to data/test_images/
"""
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import LicensePlateRecognizer, PipelineResult, PlateResult
from src.ocr_engine import format_plate_display, validate_vn_plate_format
from src.character_segmenter import segment_characters


def visualize_single_image(image_path: str, save_path: str = None):
    """Visualize the full pipeline for a single image."""
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load: {image_path}")
        return None
    
    img_name = os.path.basename(image_path)
    print(f"\n{'='*60}")
    print(f"Processing: {img_name}")
    print(f"{'='*60}")
    
    # Run recognizer with debug
    recognizer = LicensePlateRecognizer(debug=True)
    result = recognizer.recognize(img)
    
    print(f"Processing time: {result.processing_time_ms:.1f}ms")
    print(f"Plates found: {len(result.plates)}")
    
    # Create visualization
    num_plates = min(4, len(result.plates)) if result.plates else 0
    
    if num_plates == 0:
        # Just show original image if no plates found
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'{img_name}\nNo plates detected')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        return result
    
    # Create multi-row visualization
    fig = plt.figure(figsize=(16, 4 * (num_plates + 1)))
    
    # Row 1: Original image with all detections marked
    ax_main = fig.add_subplot(num_plates + 1, 1, 1)
    img_annotated = img.copy()
    
    for i, plate in enumerate(result.plates[:num_plates]):
        x, y, w, h = plate.box
        color = (0, 255, 0) if plate.confidence > 70 else (0, 200, 255)
        cv2.rectangle(img_annotated, (x, y), (x+w, y+h), color, 2)
        
        label = f"{i+1}: {plate.text}" if plate.text else f"{i+1}: ?"
        cv2.putText(img_annotated, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    ax_main.imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
    ax_main.set_title(f'{img_name[:50]}... - {len(result.plates)} plates detected', fontsize=12)
    ax_main.axis('off')
    
    # Rows 2+: Each plate detail
    for idx, plate in enumerate(result.plates[:num_plates]):
        x, y, w, h = plate.box
        roi = img[y:y+h, x:x+w]
        
        # 4 columns: ROI, Corrected, Segmented, Result
        for col in range(4):
            ax = fig.add_subplot(num_plates + 1, 4, (idx + 1) * 4 + col + 1)
            
            if col == 0:  # Original ROI
                ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                ax.set_title(f'ROI {idx+1}\n{w}x{h}', fontsize=9)
            
            elif col == 1:  # Corrected
                if plate.corrected_image is not None:
                    ax.imshow(plate.corrected_image, cmap='gray')
                    ax.set_title('Corrected', fontsize=9)
                else:
                    ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cmap='gray')
                    ax.set_title('(No correction)', fontsize=9)
            
            elif col == 2:  # Segmented characters
                if plate.segmented_chars and len(plate.segmented_chars) > 0:
                    # Create a strip of character images
                    char_strip = np.hstack([
                        cv2.resize(c, (28, 28)) for c in plate.segmented_chars[:8]
                    ])
                    ax.imshow(char_strip, cmap='gray')
                    ax.set_title(f'{len(plate.segmented_chars)} chars', fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'No chars', ha='center', va='center', fontsize=10)
                    ax.set_title('Segmentation', fontsize=9)
            
            else:  # col == 3: Result
                display_text = format_plate_display(plate.text) if plate.text else "N/A"
                is_valid, _ = validate_vn_plate_format(plate.text) if plate.text else (False, "")
                
                result_text = f"Text: {display_text}\n"
                result_text += f"Conf: {plate.confidence:.1f}%\n"
                result_text += f"Type: {plate.plate_type}\n"
                result_text += f"Valid: {'✓' if is_valid else '✗'}"
                
                ax.text(0.1, 0.5, result_text, fontsize=10, 
                       transform=ax.transAxes, verticalalignment='center',
                       family='monospace',
                       color='green' if is_valid else 'orange')
                ax.set_title('OCR Result', fontsize=9)
            
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 60)
    for i, plate in enumerate(result.plates):
        display = format_plate_display(plate.text) if plate.text else "N/A"
        is_valid, reason = validate_vn_plate_format(plate.text) if plate.text else (False, "Empty")
        
        print(f"  Plate {i+1}:")
        print(f"    Text: {display}")
        print(f"    Raw:  {plate.text}")
        print(f"    Confidence: {plate.confidence:.1f}%")
        print(f"    Type: {plate.plate_type}")
        print(f"    Box: {plate.box}")
        print(f"    Valid: {is_valid} ({reason if not is_valid else 'OK'})")
        print()
    
    return result


def visualize_batch(test_folder: str, num_images: int = 8, save_folder: str = None):
    """Visualize pipeline results for multiple images."""
    
    images = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    images = images[:num_images]
    
    print(f"\n{'='*60}")
    print("END-TO-END PIPELINE BATCH VISUALIZATION")
    print(f"{'='*60}")
    print(f"Processing {len(images)} images from {test_folder}\n")
    
    recognizer = LicensePlateRecognizer(debug=True)
    all_results = []
    
    # Create comparison figure
    fig, axes = plt.subplots(len(images), 3, figsize=(15, 4 * len(images)))
    if len(images) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_name in enumerate(images):
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Cannot load: {img_name}")
            continue
        
        result = recognizer.recognize(img)
        all_results.append(result)
        
        short_name = img_name[:25] + "..." if len(img_name) > 25 else img_name
        
        # Column 1: Original
        axes[idx, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f'{short_name}', fontsize=9)
        axes[idx, 0].axis('off')
        
        # Column 2: With detections
        img_det = img.copy()
        for plate in result.plates[:3]:
            x, y, w, h = plate.box
            color = (0, 255, 0) if plate.confidence > 70 else (0, 200, 255)
            cv2.rectangle(img_det, (x, y), (x+w, y+h), color, 2)
        
        axes[idx, 1].imshow(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f'{len(result.plates)} detections', fontsize=9)
        axes[idx, 1].axis('off')
        
        # Column 3: OCR Results
        result_text = f"Time: {result.processing_time_ms:.0f}ms\n\n"
        for i, plate in enumerate(result.plates[:3]):
            display = format_plate_display(plate.text) if plate.text else "N/A"
            is_valid, _ = validate_vn_plate_format(plate.text) if plate.text else (False, "")
            mark = "✓" if is_valid else "✗"
            result_text += f"{i+1}. {display} ({plate.confidence:.0f}%) {mark}\n"
        
        axes[idx, 2].text(0.05, 0.5, result_text, fontsize=10,
                        transform=axes[idx, 2].transAxes, 
                        verticalalignment='center',
                        family='monospace')
        axes[idx, 2].set_title('OCR Results', fontsize=9)
        axes[idx, 2].axis('off')
        
        print(f"  {short_name}: {len(result.plates)} plates, {result.processing_time_ms:.0f}ms")
    
    plt.suptitle('End-to-End License Plate Recognition Pipeline', fontsize=14)
    plt.tight_layout()
    
    if save_folder:
        save_path = os.path.join(save_folder, 'pipeline_batch_results.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved batch results to: {save_path}")
    
    plt.show()
    plt.close()
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    total_plates = sum(len(r.plates) for r in all_results)
    total_time = sum(r.processing_time_ms for r in all_results)
    avg_time = total_time / len(all_results) if all_results else 0
    
    valid_count = 0
    for r in all_results:
        for p in r.plates:
            if p.text:
                is_valid, _ = validate_vn_plate_format(p.text)
                if is_valid:
                    valid_count += 1
    
    print(f"Images processed: {len(all_results)}")
    print(f"Total plates found: {total_plates}")
    print(f"Valid format plates: {valid_count}")
    print(f"Average processing time: {avg_time:.0f}ms per image")
    print(f"Total processing time: {total_time:.0f}ms")
    
    return all_results


def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Visualize License Plate Recognition Pipeline'
    )
    parser.add_argument('--image', '-i', type=str, 
                       help='Path to single image to process')
    parser.add_argument('--folder', '-f', type=str,
                       default=r'F:\CODE\XLA\data\test_images\CarTGMT',
                       help='Folder with test images')
    parser.add_argument('--num', '-n', type=int, default=6,
                       help='Number of images to process in batch mode')
    parser.add_argument('--save', '-s', type=str,
                       default=r'F:\CODE\XLA\data\test_images',
                       help='Folder to save output images')
    
    args = parser.parse_args()
    
    if args.image:
        # Single image mode
        save_path = os.path.join(args.save, 'pipeline_single_result.png') if args.save else None
        visualize_single_image(args.image, save_path=save_path)
    else:
        # Batch mode
        if os.path.exists(args.folder):
            visualize_batch(args.folder, num_images=args.num, save_folder=args.save)
        else:
            print(f"❌ Folder not found: {args.folder}")
            return 1
    
    print(f"\n{'='*60}")
    print("✅ PIPELINE VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
