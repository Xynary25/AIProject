import os
from detect_face import detect_face
import cv2


def prepare_training_data():

    # название изображения из папки обучения
    dirs = os.listdir(r'./img_train')
    faces = []
    labels = []
    for image_path in dirs:
        # Если название картинки начинается со слова happy, метка 1; sad начинается с метки 2
        if image_path[0] == 'h':
            label = 1
        else:
            label = 2

        # путь к изображению
        image_path = './img_train/' + image_path

        # Вернуться к оттенкам серого, вернуться к объекту Mat
        image = cv2.imread(image_path,0)

        # Отображать изображение в форме окна, отображать 100 миллисекунд
        cv2.imshow("Training on image...", image)
        cv2.waitKey(100)

        face, rect = detect_face(image)
        if face is not None:
            faces.append(face)
            labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels