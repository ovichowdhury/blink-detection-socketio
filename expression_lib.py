import numpy as np
import cv2
from fer import FER
from mtcnn import MTCNN
from face_lib import base64_to_numpy

face_detector = MTCNN()

expr_detector = FER(mtcnn=True)


def detectExpression(base64_image):
    img = base64_to_numpy(base64_image)
    face = face_detector.detect_faces(img)
    if len(face) > 0:
        # print("Face Confiddence: ", face[0]['confidence'])
        box = face[0]['box']
        startX = box[0]
        startY = box[1]
        endX = startX + box[2]
        endY = startY + box[3]

        crop_img = img[startY : endY, startX : endX]

        emotion, score = expr_detector.top_emotion(crop_img)

        if score < 0.6 : 
            emotion = "neutral"
        return emotion
        
    return None
            