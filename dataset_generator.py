import cv2
import csv
from hand_tracker import HandTracker

tracker = HandTracker()

letter = input("Enter letter to record: ")

cap = cv2.VideoCapture(0)

with open("asl_dataset.csv", "a", newline="") as f:

    writer = csv.writer(f)

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        landmarks, frame = tracker.get_landmarks(frame)

        if len(landmarks) == 21:

            row = []
            for lm in landmarks:
                row.extend(lm)

            row.append(letter)

            writer.writerow(row)

            cv2.putText(frame,"Recording...",(10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Dataset Generator", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
