import cv2
from HandDetector import HandDetector
from config import CAMERA_INDEX, CAMERA_HEIGHT, CAMERA_WIDTH
from HandAnalysis import draw_land_marking

myHandDetector = HandDetector()
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit


while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    result = myHandDetector.get_hand_marking(frame)
    draw_land_marking(frame,result)


    cv2.imshow("GestureDrive", frame)

    key = cv2.waitKey(10)

    if key == 27:
        print("Closing GestureDrive...")
        break
myHandDetector.close()
cap.release()
cv2.destroyAllWindows()