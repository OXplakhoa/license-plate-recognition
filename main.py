"""CLI entry for Vietnamese License Plate Recognition."""

import argparse
from pathlib import Path

import cv2

from src.pipeline import run_pipeline, draw_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese License Plate Recognition - Traditional Image Processing",
    )
    parser.add_argument("--image", type=str, help="Path to input image (jpg, png)")
    parser.add_argument("--output", type=str, help="Path to save output image", required=False)
    parser.add_argument(
        "--plate-type",
        type=str,
        choices=["auto", "car2", "car1", "bike"],
        default="auto",
        help="Plate layout: auto (try car2, car1, bike) or force a specific type",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image:
        raise SystemExit("Please provide --image <path>")

    print("License Plate Recognition System")
    print("=" * 50)

    result = run_pipeline(args.image, plate_type=args.plate_type)

    if not result.bbox:
        print("No plate detected.")
    else:
        print(f"Detected plate: {result.text} (mode={result.mode}, conf={result.confidence:.1f}, chars={result.seg_count})")

    if args.output and result.bbox:
        img = cv2.imread(args.image)
        vis = draw_result(img, result)
        cv2.imwrite(args.output, vis)
        print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
