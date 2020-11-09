import numpy as np
import cv2
from fer import FER
from mtcnn import MTCNN


cap = cv2.VideoCapture(0)

face_detector = MTCNN()

expr_detector = FER(mtcnn=True)

while(cap.isOpened()):
    try:
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                print(f"Emotion confidence {emotion} : {score}")

                if score < 0.6 : 
                    emotion = "neutral"

                # img = cv2.rectangle(img, (10, 10), (100, 100), (255, 0, 0), 3)
                img = cv2.putText(img, str(emotion), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, None, None)
                cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.resizeWindow('Image', 600, 500)
            else:
                img = cv2.putText(img, "No Face", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, None, None)
                cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.resizeWindow('Image', 600, 500)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as ex:
        print(ex)