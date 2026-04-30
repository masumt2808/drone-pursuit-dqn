import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionState:
    vision_bit: int
    bbox: Optional[tuple]  # (cx, cy, w, h, conf), normalized


class HSVDetector:
    RED_LOWER1 = np.array([0, 40, 100])
    RED_UPPER1 = np.array([15, 255, 255])
    RED_LOWER2 = np.array([165, 40, 100])
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
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        m1 = cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1)
        m2 = cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
        mask = cv2.bitwise_or(m1, m2)

        return cv2.bitwise_and(bgr_frame, bgr_frame, mask=mask)


class YOLODetector:
    """
    YOLO bbox detector.

    For your red Crazyflie target, we first use HSV to create a target cue.
    Then YOLO returns bbox-style spatial features.

    If YOLO does not detect the drone class reliably, this class falls back
    to HSV contour bbox, which is often better for a custom red mesh in Gazebo.
    """

    def __init__(self, model_path: str = 'yolov8n.pt', conf_thresh: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.hsv = HSVDetector()

        for p in self.model.model.parameters():
            p.requires_grad = False

    def _hsv_bbox_fallback(self, bgr_frame: np.ndarray) -> VisionState:
        h, w = bgr_frame.shape[:2]
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        m1 = cv2.inRange(hsv, self.hsv.RED_LOWER1, self.hsv.RED_UPPER1)
        m2 = cv2.inRange(hsv, self.hsv.RED_LOWER2, self.hsv.RED_UPPER2)
        mask = cv2.bitwise_or(m1, m2)

        # remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        count = cv2.countNonZero(mask)
        if count < self.hsv.PIXEL_THRESHOLD:
            return VisionState(vision_bit=0, bbox=None)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return VisionState(vision_bit=0, bbox=None)

        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)

        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        bw_n = bw / w
        bh_n = bh / h

        # confidence proxy from red pixel area
        conf = float(min(1.0, count / 1000.0))

        bbox = (float(cx), float(cy), float(bw_n), float(bh_n), conf)
        return VisionState(vision_bit=1, bbox=bbox)

    def detect(self, bgr_frame: np.ndarray) -> VisionState:
        h, w = bgr_frame.shape[:2]

        hsv_result = self.hsv.detect(bgr_frame)

        # Try YOLO first
        try:
            results = self.model.predict(
                bgr_frame,
                verbose=False,
                conf=self.conf_thresh
            )

            best_bbox = None
            best_conf = 0.0

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf)

                    if conf > best_conf:
                        best_conf = conf
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        cx = ((x1 + x2) / 2.0) / w
                        cy = ((y1 + y2) / 2.0) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h

                        best_bbox = (
                            float(cx),
                            float(cy),
                            float(bw),
                            float(bh),
                            float(conf)
                        )

            if best_bbox is not None and best_conf > 0.5:
                return VisionState(vision_bit=1, bbox=best_bbox)

        except Exception:
            pass

        # Fallback is important because YOLOv8n is not trained on Crazyflie meshes.
        return self._hsv_bbox_fallback(bgr_frame)
