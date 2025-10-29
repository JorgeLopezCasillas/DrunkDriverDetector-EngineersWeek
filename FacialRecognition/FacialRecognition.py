
import cv2
import dlib
import imutils

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
            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow('Facial Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        # if press "q", "Q" or Esc to exit
        if key == ord('q') or key == ord('Q') or key == 27:  # 'q' or Esc
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
