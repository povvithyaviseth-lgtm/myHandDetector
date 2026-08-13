import cv2
from config import CAMERA_HEIGHT, CAMERA_WIDTH

# Formatting, Color: BGR
TEXT_COORD = (20,40)
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)

def draw_land_marking(frame,result):
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * CAMERA_WIDTH)
                y = int(landmark.y * CAMERA_HEIGHT)
                cv2.circle(frame, (x, y), 5, GREEN, -1)