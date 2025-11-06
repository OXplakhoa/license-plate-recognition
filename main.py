"""
Main entry point for License Plate Recognition system
Usage: python main.py --image <image_path> [--output <output_path>]
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Vietnamese License Plate Recognition - Traditional Image Processing'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image (jpg, png)',
        required=False
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save output image with detected plate',
        required=False
    )
    parser.add_argument(
        '--camera',
        action='store_true',
        help='Use camera as input (requires cv2)',
        default=False
    )
    
    args = parser.parse_args()
    
    print("License Plate Recognition System")
    print("=" * 50)
    print("\nSetup hoàn tất! Tiếp theo:")
    print("1. Cài đặt dependencies: pip install -r requirements.txt")
    print("2. Tạo Jupyter Notebook: jupyter notebook")
    print("3. Bắt đầu phát triển pipeline trong notebooks/")
    print("\nCác bước thực hiện:")
    print("  - Nhập ảnh")
    print("  - Tiền xử lý ảnh")
    print("  - Phát hiện vùng biển số")
    print("  - Phân tách ký tự")
    print("  - Nhận dạng với Tesseract")
    print("  - Hiển thị kết quả")
    
    if args.image:
        print(f"\n[TODO] Xử lý ảnh: {args.image}")
    elif args.camera:
        print("\n[TODO] Bắt đầu từ camera...")
    else:
        print("\nKhông có input. Sử dụng:")
        print("  python main.py --image <path/to/image>")
        print("  python main.py --camera")

if __name__ == "__main__":
    main()
