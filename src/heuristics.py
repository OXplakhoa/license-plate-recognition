import re

def apply_heuristics(text: str) -> str:
    """
    Corrects common OCR errors based on Vietnamese license plate rules.
    Standard Format: NN L NNNN (7 chars) or NN L NNNNN (8 chars).
    """
    if not text:
        return text

    # 1. Cleanup
    clean_text = ''.join(c for c in text if c.isalnum()).upper()
    n = len(clean_text)

    # 2. LENGTH CORRECTION (The "Ghost 1" Fix)
    # Problem: Square plates often get an inserted '1' or 'I' between the letter and numbers.
    # Example: '30E171224' (9 chars) -> Should be '30E71224'
    if n == 9:
        # Check if the structure looks like: NN L [Noise] NNNNN
        # We assume index 2 is the Series Letter (e.g., 'E')
        # We assume index 3 is the Noise (e.g., '1')
        
        likely_letter_idx_2 = clean_text[2] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" # Broad check
        likely_noise_idx_3 = clean_text[3] in ['1', 'I', 'J', 'L', '|']
        
        # If removing index 3 gives us a standard 8-char plate, do it.
        if likely_noise_idx_3:
            # Create candidate by dropping char at index 3
            candidate = clean_text[:3] + clean_text[4:]
            clean_text = candidate
            n = 8 # Update length

    # 3. MAPPINGS (Digit <-> Letter Correction)
    chars = list(clean_text)
    
    # Dictionaries for swapping
    # Confused Letter -> Should be Digit
    dict_char_to_digit = {
        'D': '0', 'O': '0', 'Q': '0',
        'I': '1', 'L': '1',
        'Z': '2',
        'J': '3',
        'A': '4',
        'S': '5',
        'G': '6',
        'T': '7',
        'B': '8',
        'Y': '4' 
    }
    
    # Confused Digit -> Should be Letter
    dict_digit_to_char = {
        '0': 'D', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'
    }

    # Rule A: First 2 characters MUST be Digits (Region Code)
    # e.g., '3OE' -> '30E'
    if n >= 2:
        for i in [0, 1]:
            if chars[i] in dict_char_to_digit:
                chars[i] = dict_char_to_digit[chars[i]]

    # Rule B: 3rd character MUST be a Letter (Series)
    # e.g., '502' -> '50Z'
    if n >= 3:
        if chars[2] in dict_digit_to_char:
            chars[2] = dict_digit_to_char[chars[2]]

    # Rule C: The rest MUST be Digits
    # e.g., '614A3' -> '61443'
    if n >= 4:
        for i in range(3, n):
            if chars[i] in dict_char_to_digit:
                chars[i] = dict_char_to_digit[chars[i]]

    return "".join(chars)

def is_valid_plate(text: str) -> bool:
    """
    Simple filter to reject garbage detections.
    """
    clean = ''.join(c for c in text if c.isalnum())
    # Valid plates are usually 7, 8, or 9 characters (some army/diplomatic are 9)
    if len(clean) < 7 or len(clean) > 9:
        return False
    return True