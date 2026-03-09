import cv2
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier

tracker = HandTracker()
classifier = GestureClassifier()

cap = cv2.VideoCapture(0)

text_output = ""

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    landmarks, frame = tracker.get_landmarks(frame)

    if len(landmarks) == 21:
        letter = classifier.predict(landmarks)

        if letter != "?":
            text_output += letter

        cv2.putText(frame, f"Letter: {letter}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

    cv2.putText(frame, f"Text: {text_output}",
                (10,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2)

    cv2.imshow("ASL Reader", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord("c"):
        text_output = ""

cap.release()
cv2.destroyAllWindows()
