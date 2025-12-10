import easyocr
import numpy as np
import cv2

# Global variable to cache the reader so we don't reload it every time
_READER = None

def get_reader():
    global _READER
    if _READER is None:
        print("Loading EasyOCR model (this might take a moment)...")
        # 'en' is usually enough for plates (numbers + letters). 
        # Add 'vi' if you need specific Vietnamese diacritics, but it might be slower.
        _READER = easyocr.Reader(['en'], gpu=False) 
    return _READER

def ocr_plate_easyocr(image: np.ndarray) -> tuple[str, float, list[float]]:
    """
    Recognizes text using EasyOCR.
    Returns: (text, confidence, list_of_char_confidences)
    """
    reader = get_reader()
    
    # EasyOCR expects RGB, OpenCV uses BGR
    if len(image.shape) == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image

    # Allowlist restricts results to Uppercase and Numbers (good for plates)
    allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # Run recognition
    results = reader.readtext(img_rgb, detail=1, allowlist=allowlist)
    
    full_text = ""
    confidences = []
    
    # results is a list of tuples: (bbox, text, prob)
    # We sort them top-to-bottom, left-to-right to handle 2-line plates correctly
    # Sorting key: (y_center // 10, x_center) to group lines roughly
    def sort_key(res):
        bbox = res[0]
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_center = (bbox[0][0] + bbox[1][0]) / 2
        return (int(y_center // 20), int(x_center))

    results.sort(key=sort_key)

    for (_, text, prob) in results:
        full_text += text
        confidences.append(prob)

    # Clean text
    clean_text = ''.join(c for c in full_text if c.isalnum()).upper()
    
    # Calculate average confidence (0.0 to 100.0)
    avg_conf = (sum(confidences) / len(confidences)) * 100 if confidences else 0.0
    
    # EasyOCR gives word-level confidence, so we just map that to list
    return clean_text, avg_conf, [avg_conf] * len(clean_text)