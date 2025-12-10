"""
End-to-End License Plate Recognition Pipeline
==============================================

This module integrates all detection, correction, segmentation, and OCR
components into a unified pipeline for Vietnamese license plate recognition.

Pipeline stages:
1. Detection: Find plate candidates using multi-preset + edge backup
2. Validation: Filter false positives using character-based scoring
3. Correction: Apply perspective transform and deskew
4. Segmentation: Extract individual character images
5. OCR: Recognize characters and apply pattern correction

Usage:
    from src.pipeline import LicensePlateRecognizer
    
    recognizer = LicensePlateRecognizer()
    results = recognizer.recognize(image)
    for result in results:
        print(f"Plate: {result.text}, Confidence: {result.confidence:.2f}")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

from .utils import ensure_grayscale
from .lp_detector import (
    detect_with_character_validation,
    detect_with_edge_backup,
    correct_plate_perspective_and_skew,
    get_plate_type_from_aspect,
)
from .character_segmenter import segment_characters, segment_characters_multi_method, SegmentationResult
from .ocr_engine import ocr_characters, ocr_plate_line, ocr_plate_multi_psm, OCRResult, configure_tesseract


@dataclass
class PlateResult:
    """Result for a single detected license plate."""
    text: str
    confidence: float
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    plate_type: str
    char_confidences: List[float] = field(default_factory=list)
    
    # Debug info
    corrected_image: Optional[np.ndarray] = None
    segmented_chars: Optional[List[np.ndarray]] = None
    detection_method: str = ""
    char_score: float = 0.0


@dataclass
class PipelineResult:
    """Result of running the full pipeline on an image."""
    plates: List[PlateResult]
    image_path: Optional[str] = None
    processing_time_ms: float = 0.0
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def best_plate(self) -> Optional[PlateResult]:
        """Return the plate with highest confidence."""
        if not self.plates:
            return None
        return max(self.plates, key=lambda p: p.confidence)
    
    @property
    def all_texts(self) -> List[str]:
        """Return all recognized plate texts."""
        return [p.text for p in self.plates if p.text]


class LicensePlateRecognizer:
    """
    End-to-end Vietnamese license plate recognizer.
    
    Combines detection, correction, segmentation, and OCR into a single interface.
    """
    
    def __init__(
        self,
        use_character_validation: bool = True,
        use_perspective_correction: bool = True,
        use_deskew: bool = True,
        min_char_score: float = 0.35,
        ocr_method: str = "segment",  # "segment" or "line"
        debug: bool = False,
    ):
        """
        Initialize the recognizer.
        
        Args:
            use_character_validation: Filter detections by character score
            use_perspective_correction: Apply perspective transform to plates
            use_deskew: Apply skew correction after perspective
            min_char_score: Minimum character score to accept detection
            ocr_method: "segment" for per-char OCR, "line" for whole-line OCR
            debug: Store debug images in results
        """
        self.use_character_validation = use_character_validation
        self.use_perspective_correction = use_perspective_correction
        self.use_deskew = use_deskew
        self.min_char_score = min_char_score
        self.ocr_method = ocr_method
        self.debug = debug
        
        # Configure Tesseract on init
        configure_tesseract()
    
    def recognize(
        self,
        image: np.ndarray,
        max_plates: int = 5,
    ) -> PipelineResult:
        """
        Recognize license plates in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            max_plates: Maximum number of plates to return
            
        Returns:
            PipelineResult containing all recognized plates
        """
        import time
        start_time = time.time()
        
        # Ensure we have both BGR and grayscale versions
        if len(image.shape) == 2:
            gray = image
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        debug_info = {}
        plates: List[PlateResult] = []
        
        # Stage 1: Detection
        if self.use_character_validation:
            detections, detect_info = detect_with_character_validation(
                gray, 
                min_char_score=self.min_char_score,
                debug=True
            )
            debug_info['detection'] = detect_info
        else:
            detections, detect_info = detect_with_edge_backup(gray, debug=True)
            debug_info['detection'] = detect_info
        
        # Limit number of detections
        detections = detections[:max_plates]
        
        # Process each detection
        for det in detections:
            plate_result = self._process_detection(gray, bgr, det)
            if plate_result and plate_result.text:
                plates.append(plate_result)
        
        # Sort by confidence
        plates.sort(key=lambda p: p.confidence, reverse=True)
        
        processing_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            plates=plates,
            processing_time_ms=processing_time,
            debug_info=debug_info,
        )
    
    def recognize_file(self, image_path: str, max_plates: int = 5) -> PipelineResult:
        """
        Recognize license plates from an image file.
        
        Args:
            image_path: Path to image file
            max_plates: Maximum number of plates to return
            
        Returns:
            PipelineResult with image_path set
        """
        image = cv2.imread(image_path)
        if image is None:
            return PipelineResult(
                plates=[],
                image_path=image_path,
                debug_info={'error': f'Cannot load image: {image_path}'}
            )
        
        result = self.recognize(image, max_plates=max_plates)
        result.image_path = image_path
        return result
    
    def _process_detection(
        self,
        gray: np.ndarray,
        bgr: np.ndarray,
        detection: Dict[str, Any],
    ) -> Optional[PlateResult]:
        """Process a single detection through correction, segmentation, and OCR."""
        
        x, y, w, h = detection['box']
        
        # Extract ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_bgr = bgr[y:y+h, x:x+w]
        
        if roi_gray.size == 0:
            return None
        
        # Determine plate type from aspect ratio
        aspect_ratio = w / float(h) if h > 0 else 1.0
        plate_type = get_plate_type_from_aspect(aspect_ratio)
        
        # Stage 2: Perspective correction
        corrected = roi_gray
        if self.use_perspective_correction:
            # The function expects full image + ROI, but we already have ROI
            # So we call with ROI covering entire extracted image
            roi_box = (0, 0, roi_gray.shape[1], roi_gray.shape[0])
            corrected_result, _ = correct_plate_perspective_and_skew(
                roi_gray,
                roi=roi_box,
                deskew=self.use_deskew
            )
            if corrected_result is not None and corrected_result.size > 0:
                corrected = corrected_result
        
        # Stage 3 & 4: Segmentation and OCR
        if self.ocr_method == "segment":
            text, confidence, char_confs, seg_chars = self._ocr_with_segmentation(
                corrected, plate_type
            )
        else:
            text, confidence, char_confs, seg_chars = self._ocr_line(
                corrected, plate_type
            )
        
        # Build result
        result = PlateResult(
            text=text,
            confidence=confidence,
            box=(x, y, w, h),
            plate_type=plate_type,
            char_confidences=char_confs,
            detection_method=detection.get('method', 'unknown'),
            char_score=detection.get('char_score', 0.0),
        )
        
        if self.debug:
            result.corrected_image = corrected.copy()
            result.segmented_chars = seg_chars
        
        return result
    
    def _ocr_with_segmentation(
        self,
        plate_image: np.ndarray,
        plate_type: str,
    ) -> Tuple[str, float, List[float], Optional[List[np.ndarray]]]:
        """OCR using character segmentation."""
        
        # Segment characters - try multiple methods
        seg_result = segment_characters_multi_method(plate_image, plate_type=plate_type)
        
        if not seg_result.char_images or len(seg_result.char_images) < 4:
            # Fallback to line OCR if segmentation fails or too few chars
            return self._ocr_line(plate_image, plate_type)
        
        # OCR each character
        ocr_result = ocr_characters(
            seg_result.char_images,
            vn_plate=True,
            plate_type=plate_type
        )
        
        text = ocr_result.text
        confidence = ocr_result.mean_conf if ocr_result.mean_conf >= 0 else 0.0
        
        return text, confidence, ocr_result.confidences, seg_result.char_images
    
    def _ocr_line(
        self,
        plate_image: np.ndarray,
        plate_type: str,
    ) -> Tuple[str, float, List[float], None]:
        """OCR using whole-line recognition with multi-PSM."""
        
        # For 2-line plates, try to split and OCR separately
        if plate_type in ['car2', 'bike']:
            text, confidence, confs = self._ocr_two_lines(plate_image, plate_type)
        else:
            # Use multi-PSM for best results
            ocr_result = ocr_plate_multi_psm(
                plate_image,
                vn_plate=True,
                plate_type=plate_type
            )
            text = ocr_result.text
            confidence = ocr_result.mean_conf if ocr_result.mean_conf >= 0 else 0.0
            confs = ocr_result.confidences
        
        return text, confidence, confs, None
    
    def _ocr_two_lines(
        self,
        plate_image: np.ndarray,
        plate_type: str,
    ) -> Tuple[str, float, List[float]]:
        """OCR a two-line plate by splitting and recognizing each line."""
        
        h, w = plate_image.shape[:2]
        
        # Split at middle with overlap
        mid_y = h // 2
        overlap = 8
        top_half = plate_image[:mid_y + overlap, :]
        bottom_half = plate_image[mid_y - overlap:, :]
        
        # OCR each half using multi-PSM
        top_result = ocr_plate_multi_psm(top_half, vn_plate=True, plate_type=plate_type)
        bottom_result = ocr_plate_multi_psm(bottom_half, vn_plate=True, plate_type=plate_type)
        
        # Combine
        text = top_result.text + bottom_result.text
        all_confs = top_result.confidences + bottom_result.confidences
        
        valid_confs = [c for c in all_confs if c >= 0]
        confidence = float(np.mean(valid_confs)) if valid_confs else 0.0
        
        return text, confidence, all_confs


def recognize_plate(
    image: np.ndarray,
    debug: bool = False,
) -> PipelineResult:
    """
    Convenience function to recognize plates in an image.
    
    Args:
        image: Input image (BGR or grayscale)
        debug: Store debug images in results
        
    Returns:
        PipelineResult containing all recognized plates
    """
    recognizer = LicensePlateRecognizer(debug=debug)
    return recognizer.recognize(image)


def recognize_plate_file(
    image_path: str,
    debug: bool = False,
) -> PipelineResult:
    """
    Convenience function to recognize plates from an image file.
    
    Args:
        image_path: Path to image file
        debug: Store debug images in results
        
    Returns:
        PipelineResult containing all recognized plates
    """
    recognizer = LicensePlateRecognizer(debug=debug)
    return recognizer.recognize_file(image_path)


# Quick test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = recognize_plate_file(image_path, debug=True)
    
    print(f"\n{'='*50}")
    print(f"Image: {image_path}")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")
    print(f"{'='*50}")
    
    if result.plates:
        for i, plate in enumerate(result.plates):
            print(f"\nPlate {i+1}:")
            print(f"  Text: {plate.text}")
            print(f"  Confidence: {plate.confidence:.1f}%")
            print(f"  Type: {plate.plate_type}")
            print(f"  Box: {plate.box}")
            print(f"  Method: {plate.detection_method}")
    else:
        print("\nNo plates detected.")
