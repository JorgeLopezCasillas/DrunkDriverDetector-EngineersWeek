
import cv2
import dlib
import imutils
import numpy as np

# Download shape_predictor_68_face_landmarks.dat from:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Place it in the same directory as this script.

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Function to calculate Eye Aspect Ratio (EAR)
    def eye_aspect_ratio(eye):
        # Calculate the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Calculate the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # Indices for the eye landmarks according to dlib's model
    LEFT_EYE_IDX = list(range(36, 42))
    RIGHT_EYE_IDX = list(range(42, 48))
    EAR_THRESHOLD = 0.21  # Typical threshold for closed eye


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=int)
            for i in range(68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
                cv2.circle(frame, (coords[i][0], coords[i][1]), 2, (0, 255, 0), -1)

            left_eye = coords[LEFT_EYE_IDX]
            right_eye = coords[RIGHT_EYE_IDX]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                eye_status = "Eyes Closed"
            else:
                eye_status = "Eyes Open"

            cv2.putText(frame, eye_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Facial Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        # Press 'q', 'Q' or Esc to exit
        if key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q' or Esc
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
