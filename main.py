import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

cv2.namedWindow("GestureDrive", cv2.WINDOW_NORMAL)

print("Camera started.")
print("ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    cv2.imshow("GestureDrive", frame)

    key = cv2.waitKey(10)

    if key == 27:
        print("Closing GestureDrive...")
        break

cap.release()
cv2.destroyAllWindows()

print("Program ended.")