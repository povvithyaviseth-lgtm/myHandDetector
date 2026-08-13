import cv2
from HandDectector import HandDetector

myHandDetector = HandDetector()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit


while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    myHandDetector.get_hand_marking(frame)

    cv2.imshow("GestureDrive", frame)

    key = cv2.waitKey(10)

    if key == 27:
        print("Closing GestureDrive...")
        break

myHandDetector.close()
cap.release()
cv2.destroyAllWindows()
