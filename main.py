#!/usr/bin/env python3
import argparse
import cv2
import os
import sys
from pathlib import Path

# Import the correct function from your pipeline
from src.pipeline import recognize_plate_file

def visualize_result(image_path, plates, output_path="output.jpg"):
    """
    Draws bounding boxes and text on the image and saves it.
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image for visualization: {image_path}")
        return

    # Loop through all detected plates
    for plate in plates:
        x, y, w, h = plate.box
        text = plate.text
        conf = plate.confidence
        
        # 1. Draw the bounding box (Green)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 2. Draw the text background (Green filled rectangle)
        label = f"{text} ({conf:.0f}%)"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), (0, 255, 0), -1)
        
        # 3. Draw the text (Black)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="License Plate Recognition System")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--plate-type", default="auto", help="Plate type (ignored, pipeline auto-detects)")
    parser.add_argument("--output", default="result.jpg", help="Path to save the output image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--engine", default="tesseract", choices=["tesseract", "easyocr"], help="OCR engine to use: tesseract(fast) or easyocr(accurate)")
    parser.add_argument("--hard-mode", action="store_true", help="Enable extra detection methods for difficult images (color + MSER)")

    args = parser.parse_args()

    # Verify file exists
    if not os.path.exists(args.image):
        print(f"Error: File not found at {args.image}")
        sys.exit(1)

    print("=" * 50)
    print(f"Processing: {args.image}")
    if args.hard_mode:
        print("Mode: HARD (color + MSER detection enabled)")
    print("=" * 50)

    # --- RUN THE PIPELINE ---
    try:
        # recognize_plate_file returns a PipelineResult object
        result = recognize_plate_file(
            args.image, 
            ocr_engine=args.engine, 
            use_color_detection=args.hard_mode,
            use_mser_detection=args.hard_mode,
            debug=args.debug
        )
        
    except Exception as e:
        print(f"Critical Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- PROCESS RESULTS ---
    if not result.plates:
        print("\nNo license plates detected.")
    else:
        print(f"\nFound {len(result.plates)} plate(s):")
        
        # Iterate through the list of detected plates
        for i, plate in enumerate(result.plates):
            print(f"\n--- Plate {i + 1} ---")
            print(f"Text       : {plate.text}")
            print(f"Confidence : {plate.confidence:.2f}%")
            print(f"Type       : {plate.plate_type}")
            print(f"Box (x,y,w,h): {plate.box}")
            
        # Draw the results on the image
        visualize_result(args.image, result.plates, args.output)

    print(f"\nTotal processing time: {result.processing_time_ms:.2f} ms")
    print("=" * 50)

if __name__ == "__main__":
    main()