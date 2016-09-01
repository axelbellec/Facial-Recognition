import os

import click
import cv2


DEFAULT_CASCADE_FOLDER = 'classifiers'
DEFAULT_FRONTAL_FACE_CLASSIFIER = 'haarcascade_frontalface_default.xml'

DEFAULT_IMG_FOLDER = 'img'
DEFAULT_GROUP_IMG = 'nasa_astronaut_group_18.jpg'


@click.command()
@click.option('--img', help='image path', type=click.Path(exists=True), default=os.path.join(DEFAULT_IMG_FOLDER, DEFAULT_GROUP_IMG))
@click.option('--face_cascade', help='frontal face classifier', type=click.Path(exists=True), default=os.path.join(DEFAULT_CASCADE_FOLDER, DEFAULT_FRONTAL_FACE_CLASSIFIER))
def main(img, face_cascade):

    cascade = cv2.CascadeClassifier(face_cascade)
    img = cv2.imread(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    face_text = '{0} face'.format(len(faces)) if len(
                faces) == 1 else '{0} faces'.format(len(faces))

    cv2.putText(img, face_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    cv2.imshow('img', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
