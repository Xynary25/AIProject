from detect_face import detect_face
from draw_rectangle import draw_rectangle
from draw_text import draw_text

def predict(test_img, face_recognizer):
    # Преобразование тегов в текст
    subjects = ['', 'Happy', 'Sad']

    # Получить копию изображения
    img = test_img.copy()

    # Определить лицо по изображению
    face, rect = detect_face(img)

    # предугадать изображение
    label = face_recognizer.predict(face)
    # Получить имя тега
    label_text = subjects[label[0]]


    # прямоугольник вокруг обнаруженного лица
    draw_rectangle(img, rect)
    # Обозначьте эмоции на лице вокруг прямоугольника
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img