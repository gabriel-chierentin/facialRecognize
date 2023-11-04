import cv2
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

facial_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

imagem = cv2.imread("exemplo.jpg")

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = detector(imagem_cinza)

for face in faces:
    pontos_faciais = predictor(imagem_cinza, face)

    vetor_de_caracteristicas = facial_recognizer.compute_face_descriptor(imagem, pontos_faciais)

    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Reconhecimento Facial", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
