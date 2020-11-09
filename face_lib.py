import face_recognition
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import cv2
from scipy.spatial import distance as dist

import os
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from mtcnn import MTCNN

# face recognition functions

TOLERANCE_THRESH = 0.475

# Global Objects

# face_detector_mtcnn = MTCNN()


class FaceNotFoundError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


def recognise(img1, img2):
    img1_encoding = face_recognition.face_encodings(img1, num_jitters=10)[0]
    img2_encoding = face_recognition.face_encodings(img2, num_jitters=10)[0]
    distance = face_recognition.face_distance([img1_encoding], img2_encoding)
    results = face_recognition.compare_faces(
        [img1_encoding], img2_encoding, tolerance=TOLERANCE_THRESH)
    return results, distance


def base64_to_numpy(base64_image):
    decoded = base64.b64decode(base64_image)
    img = np.array(Image.open(BytesIO(decoded)))
    return img

def numpy_to_base64(numpy_img):
    pil_img = Image.fromarray(numpy_img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    base64_img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64_img 


def crop_roi(img):
    face_locations = face_recognition.face_locations(img)
    top, right, bottom, left = face_locations[0]
    cropped = img[top:bottom, left:right, ::]
    return cropped

def crop_roi_extend(img, extend_height, extend_width):
    face_locations = face_recognition.face_locations(img)
    top, right, bottom, left = face_locations[0]
    top -= extend_height
    bottom += extend_height
    left -= extend_width
    right += extend_width
    cropped = img[top:bottom, left:right, ::]
    return cropped


# def crop_roi_mtcnn(img):
#     face = face_detector_mtcnn.detect_faces(img)
#     if len(face) > 0:
#         box = face[0]['box']
#         #print("Box: ", box)
#         startX = box[0]
#         startY = box[1]
#         endX = startX + box[2]
#         endY = startY + box[3]

#         roi_img_array = img[startY: endY, startX: endX]
#         return roi_img_array
#     else:
#         return None


def compare_face(base64_img1, base64_img2):
    try:
        img1 = base64_to_numpy(base64_img1)
        img2 = base64_to_numpy(base64_img2)

        # img1 = cv2.resize(img1, (250, 250))
        # img2 = cv2.resize(img2, (250, 250))

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = crop_roi(img1)
        img2 = crop_roi(img2)

        # print("after crop: ", img1.shape)
        # print("after crop: ", img2.shape)

        img1 = cv2.resize(img1, (250, 250))
        img2 = cv2.resize(img2, (250, 250))

        # cv2.imshow('i1', img1)
        # cv2.imshow('i2', img2)
        # cv2.waitKey(0)

        result, distance = recognise(img1, img2)
        return result[0], distance[0]
    except Exception as e:
        print(e)
        raise FaceNotFoundError(
            "Face recognition error please check your input", e)


def get_face_encodings(base64_img):
    try:
        img = base64_to_numpy(base64_img)
        cropped = crop_roi(img)
        encoding = face_recognition.face_encodings(cropped)[0]
        return encoding
    except Exception as e:
        raise FaceNotFoundError(
            "Face recognition error please check your input", e)


# blink detection functions

EYE_AR_THRESH = 0.25


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def avg_eye_aspect_ratio(eye1, eye2):
    avg = np.mean([eye1, eye2])
    return avg


def is_eye_open(base64_image):
    try:

        # pre processing
        image = base64_to_numpy(base64_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_roi(image)
        image = cv2.resize(image, (250, 250))

        # landmark detection
        face_landmarks_list = face_recognition.face_landmarks(image)
        rEye = eye_aspect_ratio(face_landmarks_list[0]['right_eye'])
        lEye = eye_aspect_ratio(face_landmarks_list[0]['left_eye'])
        # aspect ratio of eye calc
        avg_aspect_ratio = avg_eye_aspect_ratio(rEye, lEye)
        print("[INFO] AVG AR : ", avg_aspect_ratio)
        if avg_aspect_ratio > EYE_AR_THRESH:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        raise FaceNotFoundError(
            "Face recognition error please check your input", e)


# # face liveness detection functions
# MODEL_NAME = 'liveness_model.h5'
# LABEL_NAME = 'label.pickle'

# MODEL_PATH = os.path.abspath(
#     './face_utils/cnn_models/liveness_model/' + MODEL_NAME)
# LABEL_PATH = os.path.abspath(
#     './face_utils/cnn_models/liveness_model/' + LABEL_NAME)

# liveness_labels = pickle.loads(open(LABEL_PATH, "rb").read())
# liveness_model = load_model(MODEL_PATH)


# def is_face_live(base64_image):
#     img = base64_to_numpy(base64_image)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = crop_roi_mtcnn(img)
#     # cv2.imshow('i2', img)
#     # cv2.waitKey(0)
#     img = preprocess_input(img)
#     img = cv2.resize(img, (224, 224))
#     img = np.expand_dims(img, axis=0)
#     prediction = liveness_model.predict(img)
#     print("Predictions: ", prediction)
#     result_class = np.argmax(prediction)
#     if result_class == liveness_labels['real']:
#         print("Image is real")
#         return True
#     else:
#         print("Image is fake")
#         return False