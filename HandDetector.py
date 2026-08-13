import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from constant import *

# Formatting, Color: BGR
TEXT_COORD = (20,40)
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)

def draw_land_marking(frame, hand):
    height, width, _ = frame.shape
    for landmark in hand:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 5, GREEN, -1)


def get_fingers_up(hand):
    fingers_up = []

    if hand[THUMB_TIP].y < hand[THUMB_IP].y and  hand[THUMB_TIP].x > hand[THUMB_IP].x:
        fingers_up.append(THUMB)

    if hand[INDEX_TIP].y < hand[INDEX_PIP].y:
        fingers_up.append(INDEX)

    if hand[MIDDLE_TIP].y < hand[MIDDLE_PIP].y:
        fingers_up.append(MIDDLE)

    if hand[RING_TIP].y < hand[RING_PIP].y:
        fingers_up.append(RING)

    if hand[PINKY_TIP].y < hand[PINKY_PIP].y:
        fingers_up.append(PINKY)

    return fingers_up


class HandDetector:

    def __init__(self, num_hands=2, detection_confident=0.5, presence_confident=0.5, tracking_confidence=0.5):
        MODEL_PATH = "hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confident,
            min_hand_presence_confidence=presence_confident,
            min_tracking_confidence=tracking_confidence
        )

        self._detector = vision.HandLandmarker.create_from_options(options)

    def get_hand_marking(self,frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                      # OpenCV uses BGR
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)   # Convert numpy image into MediaPipe image
        result = self._detector.detect(mp_image)                                # Detect hand landmarks
        hands = []
        if result.hand_landmarks:
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    draw_land_marking(frame, hand)
                    hands.append(hand)
        return hands

    def close(self):
        self._detector.close()