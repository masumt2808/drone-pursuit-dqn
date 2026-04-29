import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionState:
    vision_bit: int          # 1 if target visible, 0 if not
    bbox: Optional[tuple]    # (cx, cy, w, h, conf) normalized — Option B only


class HSVDetector:
    """
    Option A — binary HSV red-pixel detector.
    Mirrors the color-thresholding approach from UAVProjectileCatcher
    but outputs a single binary bit for the DQN state vector.
    Red wraps around the HSV hue wheel so we need two ranges.
    """
    RED_LOWER1 = np.array([0,   40,  100])
    RED_UPPER1 = np.array([15,  255, 255])
    RED_LOWER2 = np.array([165,  40, 100])
    RED_UPPER2 = np.array([180, 255, 255])
    PIXEL_THRESHOLD = 22

    def detect(self, bgr_frame: np.ndarray) -> VisionState:
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1)
        m2 = cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
        mask = cv2.bitwise_or(m1, m2)
        count = cv2.countNonZero(mask)
        bit = 1 if count >= self.PIXEL_THRESHOLD else 0
        return VisionState(vision_bit=bit, bbox=None)

    def debug_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Returns masked frame for visual debugging."""
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1)
        m2 = cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
        mask = cv2.bitwise_or(m1, m2)
        return cv2.bitwise_and(bgr_frame, bgr_frame, mask=mask)


class YOLODetector:
    """
    Option B — frozen YOLOv8n spatial state extractor.
    Uses HSV as a pre-filter RoI, then runs YOLO for bbox features.
    The model is never trained — weights are frozen.
    """
    def __init__(self, model_path: str = 'yolov8n.pt'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        for p in self.model.model.parameters():
            p.requires_grad = False
        self.hsv = HSVDetector()

    def detect(self, bgr_frame: np.ndarray) -> VisionState:
        h, w = bgr_frame.shape[:2]
        # vision_bit from fast HSV check
        hsv_result = self.hsv.detect(bgr_frame)
        bit = hsv_result.vision_bit

        # YOLO for spatial bbox
        results = self.model.predict(bgr_frame, verbose=False, conf=0.3)
        best_bbox = None
        best_conf = 0.0
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    best_bbox = (cx, cy, bw, bh, conf)

        return VisionState(vision_bit=bit, bbox=best_bbox)
