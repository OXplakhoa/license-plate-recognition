"""
Integration Tests for End-to-End License Plate Recognition Pipeline
====================================================================

Tests the complete pipeline from image input to OCR output.
"""
import os
import sys
import pytest
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (
    LicensePlateRecognizer,
    PlateResult,
    PipelineResult,
    recognize_plate,
    recognize_plate_file,
)
from src.character_segmenter import segment_characters, segment_characters_multi_method
from src.ocr_engine import (
    correct_plate_by_pattern,
    smart_correct_plate,
    validate_vn_plate_format,
    format_plate_display,
)


# Test folder
TEST_FOLDER = r'F:\CODE\XLA\data\test_images\CarTGMT'


class TestOCRPostProcessing:
    """Test OCR post-processing functions."""
    
    def test_correct_plate_by_pattern(self):
        """Test basic plate pattern correction."""
        # Test digit confusion
        assert correct_plate_by_pattern("5OF12345") == "50F12345"
        assert correct_plate_by_pattern("S1F12345") == "51F12345"
        
        # Test letter confusion at position 2 - O becomes D based on _closest_letter mapping
        result = correct_plate_by_pattern("510I2345")
        assert result[2].isalpha()  # Position 2 should be a letter
        
        # Test length trimming
        result = correct_plate_by_pattern("51F123456789")
        assert len(result) == 8
    
    def test_validate_vn_plate_format(self):
        """Test Vietnamese plate format validation."""
        # Valid plates
        is_valid, _ = validate_vn_plate_format("51F12345")
        assert is_valid
        
        is_valid, _ = validate_vn_plate_format("29A12345")
        assert is_valid
        
        # Invalid - wrong length
        is_valid, reason = validate_vn_plate_format("51F12")
        assert not is_valid
        assert "Length" in reason
        
        # Invalid - unknown province
        is_valid, reason = validate_vn_plate_format("00A12345")
        assert not is_valid
        assert "province" in reason.lower()
    
    def test_smart_correct_plate(self):
        """Test smart plate correction with confidence."""
        # Valid plate
        corrected, conf = smart_correct_plate("51F12345")
        assert corrected == "51F12345"
        assert conf == 1.0
        
        # Correctable plate
        corrected, conf = smart_correct_plate("5OF12345")
        assert corrected == "50F12345"
        assert conf >= 0.5
    
    def test_format_plate_display(self):
        """Test plate display formatting."""
        assert format_plate_display("51F12345") == "51F-123.45"
        assert format_plate_display("29A12345") == "29A-123.45"
        # Short plates are returned as-is
        result = format_plate_display("51F123")
        assert "51F" in result


class TestCharacterSegmenter:
    """Test character segmentation improvements."""
    
    def test_segment_synthetic_plate(self):
        """Test segmentation on synthetic plate image."""
        # Create a simple synthetic plate image
        plate = np.ones((80, 200), dtype=np.uint8) * 255
        
        # Draw some character-like rectangles
        for i, x in enumerate([20, 45, 70, 100, 125, 150, 175]):
            cv2.rectangle(plate, (x, 20), (x + 20, 60), 0, -1)
        
        result = segment_characters(plate, plate_type="car1")
        
        # Should find some characters
        assert len(result.boxes) > 0
        assert result.binary is not None
        assert result.inverted_binary is not None
    
    def test_segment_with_different_methods(self):
        """Test that different methods work."""
        # Create test image
        plate = np.random.randint(100, 200, (80, 200), dtype=np.uint8)
        
        for method in ["otsu", "adaptive", "combined", "auto"]:
            result = segment_characters(plate, plate_type="car2", method=method)
            assert result is not None
            assert hasattr(result, 'boxes')
    
    def test_segment_multi_method(self):
        """Test multi-method segmentation."""
        plate = np.ones((80, 200), dtype=np.uint8) * 200
        
        # Draw characters
        for i, x in enumerate([20, 45, 70, 100, 125, 150]):
            cv2.rectangle(plate, (x, 15), (x + 18, 55), 30, -1)
        
        result = segment_characters_multi_method(plate, plate_type="car1")
        assert result is not None


class TestPipeline:
    """Test the full recognition pipeline."""
    
    def test_recognizer_init(self):
        """Test recognizer initialization."""
        recognizer = LicensePlateRecognizer()
        assert recognizer.use_character_validation == True
        assert recognizer.use_perspective_correction == True
        assert recognizer.min_char_score == 0.35
    
    def test_recognizer_custom_config(self):
        """Test recognizer with custom config."""
        recognizer = LicensePlateRecognizer(
            use_character_validation=False,
            use_perspective_correction=False,
            min_char_score=0.5,
            ocr_method="line",
            debug=True,
        )
        assert recognizer.use_character_validation == False
        assert recognizer.ocr_method == "line"
        assert recognizer.debug == True
    
    def test_recognize_synthetic_image(self):
        """Test recognition on synthetic image."""
        # Create a test image
        image = np.ones((400, 600, 3), dtype=np.uint8) * 180
        
        # Draw a plate-like region
        cv2.rectangle(image, (150, 150), (450, 250), (200, 200, 200), -1)
        cv2.rectangle(image, (150, 150), (450, 250), (50, 50, 50), 2)
        
        recognizer = LicensePlateRecognizer(debug=True)
        result = recognizer.recognize(image)
        
        assert isinstance(result, PipelineResult)
        assert isinstance(result.plates, list)
        assert result.processing_time_ms > 0
    
    def test_recognize_convenience_function(self):
        """Test convenience function."""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 150
        
        result = recognize_plate(image)
        assert isinstance(result, PipelineResult)
    
    @pytest.mark.skipif(
        not os.path.exists(TEST_FOLDER),
        reason="Test folder not found"
    )
    def test_recognize_real_images(self):
        """Test recognition on real images."""
        images = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith('.jpg')][:5]
        
        recognizer = LicensePlateRecognizer(debug=True)
        
        print("\n" + "=" * 60)
        print("END-TO-END PIPELINE TEST")
        print("=" * 60)
        
        results = []
        for img_name in images:
            img_path = os.path.join(TEST_FOLDER, img_name)
            result = recognizer.recognize_file(img_path)
            
            short_name = img_name[:30] + "..." if len(img_name) > 30 else img_name
            plate_texts = [p.text for p in result.plates[:3]]
            
            print(f"\n{short_name}")
            print(f"  Time: {result.processing_time_ms:.1f}ms")
            print(f"  Plates found: {len(result.plates)}")
            for i, plate in enumerate(result.plates[:3]):
                display = format_plate_display(plate.text) if plate.text else "N/A"
                print(f"    {i+1}. {display} (conf: {plate.confidence:.1f}%, type: {plate.plate_type})")
            
            results.append(result)
        
        # At least some images should have plates
        plates_found = sum(len(r.plates) for r in results)
        print(f"\nTotal plates found: {plates_found} in {len(images)} images")
        
        assert plates_found > 0, "No plates found in any test images"


class TestPlateResult:
    """Test PlateResult dataclass."""
    
    def test_plate_result_creation(self):
        """Test creating PlateResult."""
        result = PlateResult(
            text="51F12345",
            confidence=85.5,
            box=(100, 100, 200, 50),
            plate_type="car1",
        )
        
        assert result.text == "51F12345"
        assert result.confidence == 85.5
        assert result.box == (100, 100, 200, 50)
        assert result.plate_type == "car1"


class TestPipelineResult:
    """Test PipelineResult dataclass."""
    
    def test_best_plate(self):
        """Test best_plate property."""
        plates = [
            PlateResult(text="51F12345", confidence=80, box=(0, 0, 100, 50), plate_type="car1"),
            PlateResult(text="29A54321", confidence=95, box=(0, 0, 100, 50), plate_type="car2"),
        ]
        result = PipelineResult(plates=plates)
        
        best = result.best_plate
        assert best is not None
        assert best.text == "29A54321"
        assert best.confidence == 95
    
    def test_all_texts(self):
        """Test all_texts property."""
        plates = [
            PlateResult(text="51F12345", confidence=80, box=(0, 0, 100, 50), plate_type="car1"),
            PlateResult(text="29A54321", confidence=95, box=(0, 0, 100, 50), plate_type="car2"),
        ]
        result = PipelineResult(plates=plates)
        
        texts = result.all_texts
        assert len(texts) == 2
        assert "51F12345" in texts
        assert "29A54321" in texts
    
    def test_empty_result(self):
        """Test empty result."""
        result = PipelineResult(plates=[])
        
        assert result.best_plate is None
        assert result.all_texts == []


def run_all_tests():
    """Run all tests with verbose output."""
    print("\n" + "=" * 60)
    print("PIPELINE INTEGRATION TESTS")
    print("=" * 60)
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",
    ])
    
    return exit_code


if __name__ == "__main__":
    run_all_tests()
