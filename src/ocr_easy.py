import easyocr
import numpy as np
import cv2

# Global variable to cache the reader
_READER = None

def get_reader():
    global _READER
    if _READER is None:
        print("Loading EasyOCR model...")
        _READER = easyocr.Reader(['en'], gpu=False) 
    return _READER

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses the image for OCR by Upscaling and Padding.
    This helps separates details like the gap in '5' vs '6'.
    """
    # 1. Ensure image is BGR (3 channels) for consistent processing
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    h, w = image.shape[:2]
    
    # 2. Upscale Resolution (Super-Resolution Lite)
    # If the plate is small, we double its size using Bicubic interpolation.
    # This turns a 1-pixel blurry gap into a clear 2-3 pixel gap.
    if h < 300:
        scale = 2.0 
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 3. Sharpening
    # A mild sharpening kernel makes edges crisper without adding too much noise.
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    # 4. Add Padding (White Border)
    # OCR models struggle at image edges. Adding space helps context.
    pad = 20
    image = cv2.copyMakeBorder(
        image, 
        pad, pad, pad, pad, 
        cv2.BORDER_CONSTANT, 
        value=(255, 255, 255) # White background
    )

    return image

def ocr_plate_easyocr(image: np.ndarray) -> tuple[str, float, list[float]]:
    """
    Recognizes text using EasyOCR with resolution enhancement.
    """
    reader = get_reader()
    
    # --- STEP 1: ENHANCE (Upscale + Pad) ---
    enhanced_img = enhance_image(image)
    
    # Convert to RGB for EasyOCR
    img_input = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

    # Allowlist: Standard characters
    allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # --- STEP 2: RUN RECOGNITION ---
    # detail=1 gives bounding boxes
    results = reader.readtext(img_input, detail=1, allowlist=allowlist)
    
    full_text = ""
    confidences = []
    
    # Sort results top-to-bottom, then left-to-right
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
    
    # Calculate average confidence (0-100)
    avg_conf = (sum(confidences) / len(confidences)) * 100 if confidences else 0.0
    
    return clean_text, avg_conf, [avg_conf] * len(clean_text)