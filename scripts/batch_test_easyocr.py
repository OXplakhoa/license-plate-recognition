#!/usr/bin/env python
"""
Batch test license plate recognition with EasyOCR engine.
Copies successful images to success folder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import shutil
import argparse
from datetime import datetime
import cv2

from src.pipeline import LicensePlateRecognizer
from src.heuristics import is_valid_plate


def batch_test(input_dir: str, output_dir: str, engine: str = "easyocr", limit: int = None):
    """
    Batch test all images in input_dir and copy successful ones to output_dir.
    
    Args:
        input_dir: Directory containing test images
        output_dir: Directory to save successful images
        engine: OCR engine to use ("tesseract" or "easyocr")
        limit: Maximum number of images to process (None = all)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files (exclude existing result files)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions 
        and not f.name.endswith('_detailed_pipeline.png')
        and not f.name.startswith('result')
    ]
    
    if limit:
        image_files = image_files[:limit]
    
    print(f"=" * 70)
    print(f"BATCH TEST - License Plate Recognition")
    print(f"=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"OCR Engine: {engine}")
    print(f"Total images to process: {len(image_files)}")
    print(f"=" * 70)
    print()
    
    # Initialize recognizer
    recognizer = LicensePlateRecognizer(
        ocr_engine=engine,
        use_character_validation=True,
        use_perspective_correction=True,
        use_deskew=True,
        debug=False
    )
    
    # Statistics
    total = len(image_files)
    success_count = 0
    failed_count = 0
    success_files = []
    failed_files = []
    
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{total}] Processing: {img_file.name}...", end=" ")
        
        try:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print("ERROR: Cannot load image")
                failed_count += 1
                failed_files.append((img_file.name, "Cannot load image"))
                continue
            
            # Run recognition
            result = recognizer.recognize(image)
            
            if result.best_plate and result.best_plate.text:
                plate_text = result.best_plate.text
                confidence = result.best_plate.confidence
                
                # Check if valid plate format
                if is_valid_plate(plate_text):
                    print(f"SUCCESS: {plate_text} (conf: {confidence:.1f}%)")
                    success_count += 1
                    success_files.append((img_file.name, plate_text, confidence))
                    
                    # Copy to success folder
                    dest_path = output_path / img_file.name
                    shutil.copy2(img_file, dest_path)
                else:
                    print(f"INVALID FORMAT: {plate_text}")
                    failed_count += 1
                    failed_files.append((img_file.name, f"Invalid format: {plate_text}"))
            else:
                print("FAILED: No plate detected")
                failed_count += 1
                failed_files.append((img_file.name, "No plate detected"))
                
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
            failed_count += 1
            failed_files.append((img_file.name, str(e)[:50]))
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {total}")
    print(f"Successful: {success_count} ({success_count/total*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/total*100:.1f}%)")
    print()
    
    if success_files:
        print(f"Successful images copied to: {output_path}")
        print()
        print("Successful detections:")
        for filename, plate, conf in success_files:
            print(f"  {filename}: {plate} ({conf:.1f}%)")
    
    # Save report
    report_path = output_path / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Batch Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Input: {input_dir}\n")
        f.write(f"Engine: {engine}\n")
        f.write(f"Total: {total}, Success: {success_count}, Failed: {failed_count}\n")
        f.write(f"Success rate: {success_count/total*100:.1f}%\n")
        f.write(f"\n{'=' * 70}\n")
        f.write("SUCCESSFUL DETECTIONS:\n")
        f.write(f"{'=' * 70}\n")
        for filename, plate, conf in success_files:
            f.write(f"{filename}: {plate} ({conf:.1f}%)\n")
        f.write(f"\n{'=' * 70}\n")
        f.write("FAILED DETECTIONS:\n")
        f.write(f"{'=' * 70}\n")
        for filename, reason in failed_files:
            f.write(f"{filename}: {reason}\n")
    
    print(f"\nReport saved to: {report_path}")
    
    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser(description="Batch test license plate recognition")
    parser.add_argument("--input", "-i", default="data/test_images/CarTGMT",
                        help="Input directory containing images")
    parser.add_argument("--output", "-o", default="data/test_images/success",
                        help="Output directory for successful images")
    parser.add_argument("--engine", "-e", default="easyocr",
                        choices=["tesseract", "easyocr"],
                        help="OCR engine to use")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit number of images to process")
    
    args = parser.parse_args()
    
    batch_test(args.input, args.output, args.engine, args.limit)


if __name__ == "__main__":
    main()
