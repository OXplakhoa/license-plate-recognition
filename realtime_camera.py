#!/usr/bin/env python3
"""
Realtime License Plate Detection - Camera Stream (Optimized)
==============================================================

Script độc lập để nhận dạng biển số xe realtime từ webcam.
Đã tối ưu để chạy mượt với threading.

Cách chạy:
    python realtime_camera.py

Điều khiển:
    - Q: Thoát chương trình
    - S: Chụp và lưu ảnh hiện tại
    - SPACE: Tạm dừng/Tiếp tục
    - D: Bật/Tắt chế độ debug (hiện contours)
    - R: Chạy OCR trên frame hiện tại (thủ công)
    - A: Bật/Tắt Auto OCR (mặc định TẮT để mượt)

Yêu cầu:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import sys
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import LicensePlateRecognizer, EASYOCR_AVAILABLE
from src.lp_detector import detect_multi_preset

if not EASYOCR_AVAILABLE:
    print("⚠️ EasyOCR không được cài đặt. Sử dụng Tesseract.")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Cấu hình cho realtime detection."""
    # Camera
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640   # Giảm resolution để nhanh hơn
    FRAME_HEIGHT = 480
    
    # Detection
    DETECTION_INTERVAL = 0.2  # Chạy detection mỗi 200ms (5 FPS cho detection)
    AUTO_OCR = False  # Mặc định TẮT auto OCR để mượt
    OCR_ENGINE = "easyocr"  # EasyOCR chính xác hơn Tesseract
    
    # Display
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    # Colors (BGR)
    COLOR_PLATE_BOX = (0, 255, 0)
    COLOR_CANDIDATE = (0, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SUCCESS = (0, 255, 0)
    COLOR_WARNING = (0, 165, 255)


# ============================================================================
# THREADED DETECTOR
# ============================================================================

class ThreadedDetector:
    """Detector chạy trong thread riêng để không block video."""
    
    def __init__(self):
        self.ocr_engine = Config.OCR_ENGINE
        self.recognizer = None
        self._init_recognizer()
        
        # Threading
        self.frame_queue = Queue(maxsize=1)  # Chỉ giữ frame mới nhất
        self.result_queue = Queue(maxsize=1)
        self.running = False
        self.thread = None
        
        # State
        self.auto_ocr = Config.AUTO_OCR
        self.debug_mode = False
        self.last_detection_time = 0
        self.last_boxes = []  # Chỉ lưu boxes (nhanh)
        self.last_plates = []  # Lưu kết quả OCR
        self.pending_ocr = False  # Flag để trigger OCR thủ công
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
    
    def _init_recognizer(self):
        """Khởi tạo recognizer (lazy load)."""
        try:
            self.recognizer = LicensePlateRecognizer(
                ocr_engine=self.ocr_engine,
                use_character_validation=True,
                use_perspective_correction=False,  # Tắt để nhanh hơn
                use_deskew=False,
                debug=False
            )
            print(f"✓ Recognizer initialized: {self.ocr_engine}")
        except Exception as e:
            print(f"⚠️ Recognizer init failed: {e}")
            self.recognizer = None
    
    def start(self):
        """Bắt đầu thread detection."""
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("✓ Detection thread started")
    
    def stop(self):
        """Dừng thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _detection_loop(self):
        """Loop chạy trong thread riêng."""
        while self.running:
            try:
                # Lấy frame mới nhất (non-blocking)
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                current_time = time.time()
                
                # Chỉ chạy detection theo interval
                if current_time - self.last_detection_time < Config.DETECTION_INTERVAL:
                    continue
                
                self.last_detection_time = current_time
                
                # Chạy detection nhanh (chỉ tìm boxes)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # detect_multi_preset trả về (list_of_tuples, debug_info)
                detections, _ = detect_multi_preset(gray, presets=["default"])
                
                # Lọc và tính score cho các box
                boxes = []
                for box in detections:
                    # box là tuple (x, y, w, h)
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        x, y, w, h = box
                        # Tính aspect ratio để đánh giá
                        ar = w / h if h > 0 else 0
                        # Score dựa trên aspect ratio phù hợp với biển số (1.5 - 4.5)
                        if 1.0 < ar < 5.0:
                            score = 1.0 - abs(ar - 3.0) / 3.0  # Score cao nhất khi AR ~ 3
                            score = max(0.3, min(1.0, score))
                            boxes.append({
                                'box': (x, y, w, h),
                                'score': score
                            })
                
                self.last_boxes = boxes
                
                # Chạy OCR nếu auto_ocr BẬT hoặc có pending request
                if (self.auto_ocr or self.pending_ocr) and self.recognizer:
                    self.pending_ocr = False
                    try:
                        result = self.recognizer.recognize(frame)
                        if result.plates:
                            self.last_plates = result.plates
                            # In kết quả ra terminal
                            for p in result.plates:
                                print(f"✅ Detected: {p.text} ({p.confidence:.0f}%)")
                    except Exception as e:
                        print(f"OCR error: {e}")
                
            except Exception as e:
                import traceback
                print(f"Detection loop error: {e}")
                traceback.print_exc()
                time.sleep(0.5)  # Tránh spam lỗi
    
    def submit_frame(self, frame: np.ndarray):
        """Gửi frame mới để xử lý (non-blocking)."""
        # Xóa frame cũ và thêm frame mới
        try:
            self.frame_queue.get_nowait()
        except Empty:
            pass
        
        try:
            self.frame_queue.put_nowait(frame.copy())
        except:
            pass
    
    def trigger_ocr(self):
        """Trigger OCR thủ công."""
        self.pending_ocr = True
        print("📷 Đang chạy OCR...")
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ overlay lên frame (chạy trong main thread - nhanh)."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Cập nhật FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Vẽ detection boxes
        for box_info in self.last_boxes:
            x, y, bw, bh = box_info['box']
            score = box_info['score']
            color = Config.COLOR_SUCCESS if score >= 0.5 else Config.COLOR_CANDIDATE
            cv2.rectangle(display, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(display, f"{score:.2f}", (x, y - 5), 
                       Config.FONT, 0.5, color, 1)
        
        # Vẽ kết quả OCR (nếu có)
        for plate in self.last_plates:
            x, y, bw, bh = plate.box
            cv2.rectangle(display, (x, y), (x + bw, y + bh), Config.COLOR_PLATE_BOX, 3)
            if plate.text:
                label = f"{plate.text} ({plate.confidence:.0f}%)"
                # Background
                (tw, th), _ = cv2.getTextSize(label, Config.FONT, 0.7, 2)
                cv2.rectangle(display, (x, y - th - 10), (x + tw + 10, y), Config.COLOR_PLATE_BOX, -1)
                cv2.putText(display, label, (x + 5, y - 5), Config.FONT, 0.7, (0, 0, 0), 2)
        
        # HUD - Top bar
        cv2.rectangle(display, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(display, "License Plate Detection", (10, 25), 
                   Config.FONT, 0.7, Config.COLOR_TEXT, 2)
        
        # Status line
        status = f"FPS: {self.fps:.0f}"
        status += f" | Auto-OCR: {'ON' if self.auto_ocr else 'OFF'}"
        if self.debug_mode:
            status += " | DEBUG"
        cv2.putText(display, status, (10, 50), Config.FONT, 0.5, Config.COLOR_WARNING, 1)
        
        # Detected plates panel (right side)
        if self.last_plates:
            texts = [p.text for p in self.last_plates if p.text]
            if texts:
                panel_text = " | ".join(texts)
                cv2.rectangle(display, (w - 350, 70), (w - 10, 110), (0, 80, 0), -1)
                cv2.putText(display, panel_text, (w - 340, 95), 
                           Config.FONT, 0.7, Config.COLOR_SUCCESS, 2)
        
        # Help text - Bottom
        help_text = "Q:Quit | R:Run OCR | A:Auto-OCR | S:Save | D:Debug"
        cv2.putText(display, help_text, (10, h - 10), Config.FONT, 0.4, (120, 120, 120), 1)
        
        return display


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Hàm chính."""
    print("\n" + "=" * 60)
    print("🚗 REALTIME LICENSE PLATE DETECTION (Optimized)")
    print("=" * 60)
    
    # Khởi tạo detector
    print("\n📦 Đang khởi tạo...")
    detector = ThreadedDetector()
    
    # Mở camera
    print(f"📷 Đang mở camera {Config.CAMERA_INDEX}...")
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return
    
    # Cấu hình camera - resolution thấp để nhanh
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm lag
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera: {actual_w}x{actual_h}")
    
    # Bắt đầu detection thread
    detector.start()
    
    print("\n" + "-" * 60)
    print("🎮 ĐIỀU KHIỂN:")
    print("   Q     : Thoát")
    print("   R     : Chạy OCR (thủ công)")
    print("   A     : Bật/Tắt Auto-OCR")
    print("   S     : Lưu ảnh")
    print("   D     : Debug mode")
    print("-" * 60)
    print("\n🚀 Đang chạy... (Auto-OCR đang TẮT để mượt)")
    print("   Nhấn 'R' để OCR | 'A' để bật Auto-OCR\n")
    
    # Window
    window_name = "License Plate Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    
    paused = False
    current_frame = None
    
    try:
        while True:
            # Đọc frame
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera read failed")
                    break
                current_frame = frame
                
                # Gửi frame cho detection thread
                detector.submit_frame(frame)
            
            if current_frame is None:
                continue
            
            # Vẽ overlay (nhanh, chạy trong main thread)
            display = detector.draw_overlay(current_frame)
            
            # Hiển thị
            cv2.imshow(window_name, display)
            
            # Xử lý phím - waitKey(1) để responsive
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            elif key == ord('r') or key == ord('R'):
                detector.trigger_ocr()
            
            elif key == ord('a') or key == ord('A'):
                detector.auto_ocr = not detector.auto_ocr
                status = "BẬT" if detector.auto_ocr else "TẮT"
                print(f"🔄 Auto-OCR: {status}")
            
            elif key == ord('s') or key == ord('S'):
                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path("debug_output/snapshots")
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = output_dir / f"snapshot_{timestamp}.jpg"
                cv2.imwrite(str(filename), display)
                print(f"✓ Saved: {filename}")
            
            elif key == ord('d') or key == ord('D'):
                detector.debug_mode = not detector.debug_mode
            
            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸️ PAUSED' if paused else '▶️ RESUMED'}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    
    finally:
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup done")


if __name__ == "__main__":
    main()
