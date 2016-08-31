import os
import sys

import click
import cv2

DEFAULT_CASCADE_FOLDER = 'classifiers'
DEFAULT_FRONTAL_FACE_CLASSIFIER = 'haarcascade_frontalface_default.xml'
DEFAULT_EYE_CLASSIFIER = 'haarcascade_eyes.xml'


def add_eyes_rect(gray, frame, eye_cascade, x, y, w, h):
    eye_gray = gray[y:y + h, x:x + w]
    eye_color = frame[y:y + h, x:x + w]
    eyes_data = eye_cascade.detectMultiScale(eye_gray)

    for (ex, ey, ew, eh) in eyes_data:
        cv2.rectangle(img=eye_color,
                      pt1=(ex, ey),
                      pt2=(ex + ew, ey + eh),
                      color=(255, 255, 0),
                      thickness=2)

    eyes_text = '{} eye'.format(len(eyes_data)) if len(
        eyes_data) == 1 else '{} eyes'.format(len(eyes_data))
    return ' - ' + eyes_text


@click.command()
@click.option('--eyes', help='add eyes detection', type=bool, default=False)
@click.option('--face_cascade', help='frontal face classifier', type=click.Path(exists=True), default=os.path.join(DEFAULT_CASCADE_FOLDER, DEFAULT_FRONTAL_FACE_CLASSIFIER))
@click.option('--eyes_cascade', help='eyes classifier', type=click.Path(exists=True), default=os.path.join(DEFAULT_CASCADE_FOLDER, DEFAULT_EYE_CLASSIFIER))
def main(eyes, face_cascade, eyes_cascade):
    # Define cascade classifiers
    face_cascade = cv2.CascadeClassifier(face_cascade)
    eye_cascade = cv2.CascadeClassifier(eyes_cascade)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_data = face_cascade.detectMultiScale(gray, 1.2, 5)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces_data:
            cv2.rectangle(img=frame,
                          pt1=(x, y),
                          pt2=(x + w, y + h),
                          color=(255, 0, 0),
                          thickness=2)

            face_text = '{} face'.format(len(faces_data)) if len(
                faces_data) == 1 else '{} faces'.format(len(faces_data))

            # If we want to see the eyes
            if eyes:
                face_text += add_eyes_rect(gray,
                                           frame,
                                           eye_cascade,
                                           x, y, w, h)

        # Display the number of faces detected
        cv2.putText(img=frame,
                    text=face_text,
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=255)

        # Display the resulting frame
        cv2.imshow('Face Detection using a webcam ', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
