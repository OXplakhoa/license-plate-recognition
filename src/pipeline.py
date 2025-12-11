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
from .heuristics import apply_heuristics, is_valid_plate 

# Conditional import for EasyOCR (may not be installed)
EASYOCR_AVAILABLE = False
ocr_plate_easyocr = None
try:
    from .ocr_easy import ocr_plate_easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass 

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
        ocr_engine: str = "tesseract",  # "tesseract" or "easyocr"
        use_color_detection: bool = False,  # Use color-based detection
        use_mser_detection: bool = False,  # Use MSER text detection
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
            use_color_detection: Use color-based detection for difficult images
            use_mser_detection: Use MSER text region detection for difficult images
            debug: Store debug images in results
        """
        self.use_character_validation = use_character_validation
        self.use_perspective_correction = use_perspective_correction
        self.use_deskew = use_deskew
        self.min_char_score = min_char_score
        self.ocr_method = ocr_method
        self.ocr_engine = ocr_engine
        self.use_color_detection = use_color_detection
        self.use_mser_detection = use_mser_detection
        self.debug = debug
        
        # Only configure Tesseract if using it
        if self.ocr_engine == "tesseract":
            configure_tesseract()
    
    def _is_plate_like_image(self, image: np.ndarray) -> bool:
        """
        Check if the input image looks like a cropped license plate.
        
        This detects cases where user uploads a pre-cropped plate image
        instead of a full car photo. In such cases, we should try direct
        OCR instead of running detection.
        
        Criteria:
        - Aspect ratio matches plate types (0.7-7.0)
        - Image is relatively small (not a full car photo)
        - Has high text density (many edge pixels)
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return False
            
        aspect_ratio = w / h
        
        # Check aspect ratio matches plate types
        # Square plate: 0.7-2.2, Rectangular: 2.5-7.0, Bike: 1.0-3.2
        if not (0.5 <= aspect_ratio <= 8.0):
            return False
        
        # If image is large, it's probably a full photo, not a cropped plate
        # Typical cropped plates are small (height < 300px usually)
        # A full car photo is typically > 400px in both dimensions
        total_pixels = w * h
        if total_pixels > 200000:  # > ~450x450 
            return False
        
        # Very small images are also not plates (icons, thumbnails)
        if h < 30 or w < 50:
            return False
        
        # Check for text-like content using edge density
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (w * h)
        
        # Plate images typically have edge density between 5-40%
        # Full car photos have much lower edge density in general
        return 0.03 <= edge_density <= 0.50
    
    def _try_direct_ocr(self, image: np.ndarray, debug_info: dict) -> Optional[PlateResult]:
        """
        Try OCR directly on the input image (assuming it's a cropped plate).
        
        This is used when the input image appears to be a pre-cropped plate.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            bgr = image
        else:
            gray = image
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        h, w = gray.shape[:2]
        
        # Resize for better OCR - Tesseract needs larger images
        if self.ocr_engine == "tesseract":
            # Tesseract works better with height around 200-300px
            target_height = 250
            if h < target_height:
                scale = target_height / h
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            # EasyOCR doesn't need as much upscaling
            if h < 50:
                scale = 50 / h
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Try OCR
        if self.ocr_engine == "easyocr" and EASYOCR_AVAILABLE and ocr_plate_easyocr:
            text, confidence, _ = ocr_plate_easyocr(bgr)  # EasyOCR prefers BGR
        else:
            # Tesseract: try multiple preprocessing approaches
            best_text = ""
            best_conf = 0.0
            
            # Method 1: CLAHE enhanced
            ocr_result = ocr_plate_multi_psm(enhanced)
            if ocr_result.text and len(ocr_result.text) >= 4:
                best_text = ocr_result.text
                best_conf = ocr_result.mean_conf if ocr_result.mean_conf >= 0 else 50.0
            
            # Method 2: Otsu binarization if Method 1 failed
            if len(best_text) < 4:
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ocr_result2 = ocr_plate_multi_psm(binary)
                if ocr_result2.text and len(ocr_result2.text) > len(best_text):
                    best_text = ocr_result2.text
                    best_conf = ocr_result2.mean_conf if ocr_result2.mean_conf >= 0 else 50.0
            
            # Method 3: Adaptive threshold if still not good
            if len(best_text) < 4:
                adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                ocr_result3 = ocr_plate_multi_psm(adaptive)
                if ocr_result3.text and len(ocr_result3.text) > len(best_text):
                    best_text = ocr_result3.text
                    best_conf = ocr_result3.mean_conf if ocr_result3.mean_conf >= 0 else 50.0
            
            text = best_text
            confidence = best_conf
        
        # Apply heuristics
        if text:
            text = apply_heuristics(text)
        
        debug_info['direct_ocr'] = {
            'attempted': True,
            'raw_text': text,
            'confidence': confidence
        }
        
        # Validate result - lower threshold for Tesseract since confidence may be unreliable
        min_confidence = 20 if self.ocr_engine == "tesseract" else 30
        if text and len(text) >= 5 and confidence >= min_confidence:
            aspect_ratio = w / h
            plate_type = get_plate_type_from_aspect(aspect_ratio)
            
            return PlateResult(
                text=text,
                confidence=confidence,
                box=(0, 0, w, h),
                plate_type=plate_type,
                detection_method="direct_ocr",
                corrected_image=enhanced,
            )
        
        return None
    
    def _detect_with_retry(
        self, 
        gray: np.ndarray, 
        bgr: np.ndarray,
        debug_info: dict
    ) -> List[dict]:
        """
        Detect plates with automatic retry using additional methods if initial detection fails.
        
        Strategy:
        1. Try standard detection (contour + edge backup)
        2. If no valid detection found, retry with color detection
        3. If still no valid detection, retry with MSER
        
        This allows processing most images quickly while falling back to
        more expensive methods only when needed.
        """
        # Method 1: Standard detection (fast)
        if self.use_character_validation:
            detections, detect_info = detect_with_character_validation(
                gray,
                bgr_image=bgr,
                min_char_score=self.min_char_score,
                use_color=self.use_color_detection,
                use_mser=self.use_mser_detection,
                debug=True
            )
            debug_info['detection'] = detect_info
        else:
            detections, detect_info = detect_with_edge_backup(gray, debug=True)
            debug_info['detection'] = detect_info
        
        # Check if we have valid detections
        valid_detections = [d for d in detections if d.get("char_valid", False)]
        
        # If no valid detections and we haven't used color/MSER, try them
        if not valid_detections and not (self.use_color_detection or self.use_mser_detection):
            debug_info['retry'] = []
            
            # Retry with color detection (using BGR image)
            retry_detections, retry_info = detect_with_character_validation(
                gray,
                bgr_image=bgr,
                min_char_score=self.min_char_score * 0.8,  # Lower threshold for retry
                use_color=True,
                use_mser=False,
                debug=True
            )
            debug_info['retry'].append({'method': 'color', 'info': retry_info})
            
            valid_retry = [d for d in retry_detections if d.get("char_valid", False)]
            if valid_retry:
                detections = retry_detections
                debug_info['detection_method'] = 'color_retry'
            else:
                # Retry with MSER detection
                retry_detections, retry_info = detect_with_character_validation(
                    gray,
                    bgr_image=bgr,
                    min_char_score=self.min_char_score * 0.8,
                    use_color=False,
                    use_mser=True,
                    debug=True
                )
                debug_info['retry'].append({'method': 'mser', 'info': retry_info})
                
                valid_retry = [d for d in retry_detections if d.get("char_valid", False)]
                if valid_retry:
                    detections = retry_detections
                    debug_info['detection_method'] = 'mser_retry'
                else:
                    # Final attempt: use both color + MSER
                    retry_detections, retry_info = detect_with_character_validation(
                        gray,
                        bgr_image=bgr,
                        min_char_score=self.min_char_score * 0.7,
                        use_color=True,
                        use_mser=True,
                        debug=True
                    )
                    debug_info['retry'].append({'method': 'color_mser', 'info': retry_info})
                    
                    if retry_detections:
                        detections = retry_detections
                        debug_info['detection_method'] = 'color_mser_retry'
        
        return detections
    
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
        
        # Check if image looks like a pre-cropped plate
        # If so, try direct OCR first before running detection
        is_plate_like = self._is_plate_like_image(image)
        debug_info['is_plate_like'] = is_plate_like
        
        if is_plate_like:
            # Try direct OCR on the input image
            direct_result = self._try_direct_ocr(image, debug_info)
            if direct_result and direct_result.confidence > 50:
                plates.append(direct_result)
                # Return early if we got a good result
                processing_time = (time.time() - start_time) * 1000
                return PipelineResult(
                    plates=plates,
                    processing_time_ms=processing_time,
                    debug_info=debug_info,
                )
        
        # Stage 1: Detection with optional auto-retry
        detections = self._detect_with_retry(gray, bgr, debug_info)
        
        # Limit number of detections
        detections = detections[:max_plates]
        
        # Process each detection
        for det in detections:
            plate_result = self._process_detection(gray, bgr, det)
            if plate_result and plate_result.text:
                plates.append(plate_result)
        
        # If no plates found and we haven't tried color/MSER, retry with enhanced detection
        if not plates and not (self.use_color_detection or self.use_mser_detection):
            debug_info['ocr_retry'] = True
            
            # Try with color detection
            color_detections, color_info = detect_with_character_validation(
                gray,
                bgr_image=bgr,
                min_char_score=self.min_char_score * 0.6,  # Lower threshold
                use_color=True,
                use_mser=True,
                debug=True
            )
            debug_info['color_mser_detection'] = color_info
            
            # Sort by char_score and process
            color_detections.sort(key=lambda d: d.get('char_score', 0), reverse=True)
            for det in color_detections[:max_plates]:
                if det.get('char_valid', False):
                    plate_result = self._process_detection(gray, bgr, det)
                    if plate_result and plate_result.text:
                        plates.append(plate_result)
        
        # Final fallback: if still no plates and image looks like a plate, try direct OCR
        if not plates and is_plate_like:
            direct_result = self._try_direct_ocr(image, debug_info)
            if direct_result:
                plates.append(direct_result)
                debug_info['fallback_direct_ocr'] = True
        
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
        
        # Determine plate type from aspect ratio AND size
        aspect_ratio = w / float(h) if h > 0 else 1.0
        plate_type = get_plate_type_from_aspect(aspect_ratio, w, h)
        
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
        
        # Upscale small ROIs for better OCR (especially for rectangular plates)
        # Tesseract works better with images at least 50-60 pixels tall
        MIN_HEIGHT_FOR_OCR = 50
        if corrected.shape[0] < MIN_HEIGHT_FOR_OCR:
            scale_factor = MIN_HEIGHT_FOR_OCR / corrected.shape[0]
            # Use INTER_CUBIC for upscaling (better quality than INTER_LINEAR)
            corrected = cv2.resize(corrected, None, fx=scale_factor, fy=scale_factor, 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for better contrast (helps OCR especially on low-contrast plates)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        corrected = clahe.apply(corrected)
        
        # Stage 3 & 4: Segmentation and OCR
        if self.ocr_engine == "easyocr" and EASYOCR_AVAILABLE and ocr_plate_easyocr:
            raw_text, confidence, char_confs = ocr_plate_easyocr(corrected)
            seg_chars = None
        else:
            # Tesseract path
            if self.ocr_method == "segment":
                raw_text, confidence, char_confs, seg_chars = self._ocr_with_segmentation(corrected, plate_type)
            else:
                raw_text, confidence, char_confs, seg_chars = self._ocr_line(corrected, plate_type)

        # --- NEW: APPLY HEURISTICS ---
        # 1. Correct the text (e.g. 502... -> 50Z...)
        final_text = apply_heuristics(raw_text)
        
        # 2. Filter invalid plates (e.g. "507")
        if not is_valid_plate(final_text):
            # If initial OCR looks invalid, try segmentation-based OCR as a fallback
            try:
                seg_result = segment_characters_multi_method(corrected, plate_type=plate_type)
                if seg_result and getattr(seg_result, 'char_images', None):
                    from .ocr_engine import ocr_characters
                    seg_ocr = ocr_characters(seg_result.char_images, vn_plate=True, plate_type=plate_type)
                    seg_text = apply_heuristics(seg_ocr.text)
                    if is_valid_plate(seg_text):
                        final_text = seg_text
                        confidence = seg_ocr.mean_conf if seg_ocr.mean_conf >= 0 else confidence
                    else:
                        return None
                else:
                    return None
            except Exception:
                return None

        # Build result with the CORRECTED text
        result = PlateResult(
            text=final_text,  # Use final_text instead of text/raw_text
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
    ocr_engine: str = "tesseract",
    use_color_detection: bool = False,
    use_mser_detection: bool = False,
    debug: bool = False,
) -> PipelineResult:
    """
    Convenience function to recognize plates from an image file.
    
    Args:
        image_path: Path to image file
        ocr_engine: OCR engine ("tesseract" or "easyocr")
        use_color_detection: Use color-based detection for difficult images
        use_mser_detection: Use MSER text region detection for difficult images
        debug: Store debug images in results
        
    Returns:
        PipelineResult containing all recognized plates
    """
    recognizer = LicensePlateRecognizer(
        debug=debug, 
        ocr_engine=ocr_engine,
        use_color_detection=use_color_detection,
        use_mser_detection=use_mser_detection,
    )
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
