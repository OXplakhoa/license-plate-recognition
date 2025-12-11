import cv2
import sys
sys.path.insert(0, '.')

from app import visualize_preprocessing_steps

img = cv2.imread('data/test_images/CarTGMT/AEONTP_5026788_checkin_2020-1-13-16-19_f7zXETONv.jpg')
steps = visualize_preprocessing_steps(img)

print(f"Candidates: {steps['candidate_count']}")
print(f"Scored: {len(steps.get('scored_list', []))}")
print("\nCharacter Score cho từng candidate:")
for c in steps.get('scored_list', [])[:8]:
    status = "✓ VALID" if c['is_valid'] else "✗ reject"
    print(f"  #{c['index']:2d}: score={c['char_score']:.3f}, chars={c['char_count']}, {status}")
