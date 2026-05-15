import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionState:
    vision_bit: int
    bbox: Optional[tuple] = None


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
