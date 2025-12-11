#!/usr/bin/env python3
import sys
from pathlib import Path
# Ensure project root is on sys.path so `src` imports work when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import LicensePlateRecognizer
from src.lp_detector import detect_with_character_validation
import cv2
from src.lp_detector import correct_plate_perspective_and_skew
from src.ocr_easy import ocr_plate_easyocr
from src.ocr_engine import ocr_plate_multi_psm
from src.heuristics import apply_heuristics, is_valid_plate
from src.character_segmenter import segment_characters, segment_characters_multi_method
from src.ocr_engine import ocr_characters

if len(sys.argv) < 2:
    print("Usage: python scripts/debug_detect_tune.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
print(f"Running tuned recognizer on: {image_path}")
rec = LicensePlateRecognizer(
    use_character_validation=True,
    use_perspective_correction=True,
    use_deskew=True,
    min_char_score=0.20,
    ocr_engine='easyocr',
    use_color_detection=True,
    use_mser_detection=True,
    debug=True,
)
res = rec.recognize_file(image_path)
print(f"Processing time: {res.processing_time_ms:.1f} ms")
print(f"Plates found: {len(res.plates)}")
for i,p in enumerate(res.plates):
    print(f"-- Plate {i+1} --")
    print(f" text: {p.text}")
    print(f" conf: {p.confidence}")
    print(f" box : {p.box}")
    print(f" method: {p.detection_method}")

print('\nFull debug_info:')
import json
try:
    print(json.dumps(res.debug_info, default=str, indent=2))
except Exception:
    print(repr(res.debug_info))

# Save debug images if present
out_dir = Path(image_path).parent
if res.plates and res.plates[0].corrected_image is not None:
    out = out_dir / (Path(image_path).stem + '_corrected.png')
    import cv2
    cv2.imwrite(str(out), res.plates[0].corrected_image)
    print('Saved corrected image to', out)

print('Done')

# Directly run detection to inspect candidates and OCR per-candidate
img = cv2.imread(image_path)
if img is None:
    print('Cannot load image for detailed debug')
    sys.exit(1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print('\nRunning detect_with_character_validation directly (min_char_score=0.20, color+mser)')
cands, dinfo = detect_with_character_validation(gray, bgr_image=img, use_color=True, use_mser=True, min_char_score=0.20, debug=True)
print('Candidates count:', len(cands))
for i, c in enumerate(cands):
    print(f' Candidate {i}: box={c.get("box")}, method={c.get("method")}, char_valid={c.get("char_valid")}, char_score={c.get("char_score")}, extra_keys={[k for k in c.keys() if k not in ("box","method","char_valid","char_score")] }')

print('\nProcessing each candidate through _process_detection to see OCR outputs:')
for i, c in enumerate(cands):
    pr = rec._process_detection(gray, img, c)
    print(f' -> Candidate {i} processed ->', 'PlateResult' if pr else 'Rejected')
    if pr:
        print('    text:', pr.text)
        print('    conf:', pr.confidence)
        print('    box :', pr.box)
        print('    detection_method:', pr.detection_method)
    else:
        # Inspect OCR for this candidate
        box = c.get('box')
        if box:
            x,y,w,h = map(int, box)
            roi = gray[y:y+h, x:x+w]
            corr, info = correct_plate_perspective_and_skew(roi, roi=(0,0,roi.shape[1], roi.shape[0]), deskew=True)
            if corr is None:
                corr = roi
            # convert to BGR for easyocr
            corr_bgr = cv2.cvtColor(corr, cv2.COLOR_GRAY2BGR) if len(corr.shape)==2 else corr
            raw_text, conf, confs = ocr_plate_easyocr(corr_bgr)
            # Also try Tesseract multi-psm
            t_result = ocr_plate_multi_psm(corr, vn_plate=True)
            t_text = t_result.text
            t_conf = t_result.mean_conf
            print(f"    Tesseract raw: '{t_text}' conf={t_conf:.1f}%")
            heur = apply_heuristics(raw_text)
            valid = is_valid_plate(heur)
            print(f"    OCR raw: '{raw_text}' conf={conf:.1f}% -> heur: '{heur}' valid={valid}")
            # Try segmentation-based OCR (multi-method)
            seg = segment_characters_multi_method(corr, plate_type=None)
            if seg and getattr(seg, 'char_images', None):
                print('    Segmentation produced', len(seg.char_images), 'char images')
                ocr_res = ocr_characters(seg.char_images, vn_plate=True)
                print('    Seg-OCR:', ocr_res.text, 'conf_mean:', ocr_res.mean_conf)
