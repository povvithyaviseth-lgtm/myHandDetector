import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from constant import *
from config import MODEL_PATH, NUM_HANDS, DETECTION_CONFIDENCE, PRESENCE_CONFIDENCE, TRACKING_CONFIDENCE, CAMERA_HEIGHT, CAMERA_WIDTH


class HandDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_hand_presence_confidence=PRESENCE_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )

        self._detector = vision.HandLandmarker.create_from_options(options)

    def get_hand_marking(self,frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                      # OpenCV uses BGR
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)   # Convert numpy image into MediaPipe image
        result = self._detector.detect(mp_image)                                # Detect hand landmarks
        return result

    def close(self):
        self._detector.close()