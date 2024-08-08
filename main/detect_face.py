import cv2


def detect_face(img):

    # Преобразование изображения в серые оттенки
    gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)

    # Загрузите распознаватель лиц
    face_cascade = cv2.CascadeClassifier(r'lbpcascade_frontalface.xml')

    # Здесь выберите размер изображения
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)

    # Если на картинке нет лица
    if len(faces) == 0:
        return None, None

    # Извлечь область лица
    (x, y, w, h) = faces[0]

    # Вернуться к лицу и его области
    return gray[y:y + w, x:x + h], faces[0]