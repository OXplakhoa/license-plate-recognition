import re

# Valid Vietnamese province codes (2 digits, 11-99 and some special)
# Source: Vietnamese license plate format
VALID_PROVINCE_CODES = {
    # Hanoi
    '29', '30', '31', '32', '33', '40',
    # HCM
    '41', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
    # Other provinces
    '11', '12', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28',
    '34', '35', '36', '37', '38', '39',
    '43', '47', '48', '49',
    '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
    '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',
    '80', '81', '82', '83', '84', '85', '86', '88', '89',
    '90', '91', '92', '93', '94', '95', '97', '98', '99',
    # Special codes
    '42', '44', '45', '46', '87', '96',
}

# Common OCR confusion pairs for province codes
PROVINCE_CODE_FIXES = {
    '00': '30',  # Very common: 0 <-> 3 confusion
    '01': '31',
    '02': '32',
    '03': '33',
    '08': '38',
    '09': '39',
    '04': '34', 
    '05': '35',
    '06': '36',
    '07': '37',
    '10': '10',  # Keep as-is if invalid, might be 17/18/19
}

def fix_province_code(code: str) -> str:
    """Fix common OCR errors in province codes."""
    if code in VALID_PROVINCE_CODES:
        return code
    
    # Check specific fixes
    if code in PROVINCE_CODE_FIXES:
        fixed = PROVINCE_CODE_FIXES[code]
        if fixed in VALID_PROVINCE_CODES:
            return fixed
    
    # Try common single-character fixes
    # 0 <-> 3, 0 <-> 8, 0 <-> 9
    alternatives = []
    
    # First digit fixes
    d0, d1 = code[0], code[1]
    for alt0 in [d0, '3' if d0 == '0' else '0' if d0 == '3' else d0,
                 '8' if d0 == '0' else d0, '9' if d0 == '0' else d0]:
        for alt1 in [d1, '3' if d1 == '0' else '0' if d1 == '3' else d1,
                     '8' if d1 == '0' else d1, '9' if d1 == '0' else d1]:
            alt_code = alt0 + alt1
            if alt_code in VALID_PROVINCE_CODES and alt_code != code:
                alternatives.append(alt_code)
    
    if alternatives:
        return alternatives[0]  # Return first valid alternative
    
    return code  # Keep original if no fix found

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

    # Rule D: Validate and fix province code (first 2 digits)
    # e.g., '00F25697' -> '30F25697' (00 is not valid, 30 is Hanoi)
    if n >= 2:
        province_code = chars[0] + chars[1]
        fixed_code = fix_province_code(province_code)
        if fixed_code != province_code:
            chars[0] = fixed_code[0]
            chars[1] = fixed_code[1]

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