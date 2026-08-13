import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:

    def __init__(self):
        MODEL_PATH = "hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self._detector = vision.HandLandmarker.create_from_options(options)

    def get_hand_marking(self,frame):
        # Formatting, Color: BGR
        text_coord = (20,40)
        font = cv2.FONT_HERSHEY_SIMPLEX
        red = (0, 0, 255)
        green = (0, 255, 0)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                      # OpenCV uses BGR
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)   # Convert numpy image into MediaPipe image
        result = self._detector.detect(mp_image)                                # Detect hand landmarks
        if result.hand_landmarks:
            cv2.putText(frame, "HAND DETECTED", text_coord, font, 1, green, 2)
            hand = result.hand_landmarks[0]                                     # Get first detected hand
            height, width, _ = frame.shape
            # Draw each landmark
            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 5, green, -1)
        else:
            cv2.putText(frame, "NO HAND", text_coord, font, 1, red, 2)


    def close(self):
        self._detector.close()