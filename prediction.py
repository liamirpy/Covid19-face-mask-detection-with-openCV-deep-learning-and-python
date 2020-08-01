from tensorflow.keras.models import load_model
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
model_mask = load_model('mask_nomask.h5')


def detect_face(image):

    face_img=image.copy()
    face_rects=face_cascade.detectMultiScale(face_img)
    for (x,y,w,h) in face_rects:

        F=face_img[y:y+w,x:x+w]
        img = cv2.resize(F, (100, 100))
        imag_ = np.zeros((1, 100, 100, 3), dtype=np.uint16)
        for i in range(100):
            for j in range(100):
                imag_[0, i, j, 0] = img[i, j, 0]
                imag_[0, i, j, 1] = img[i, j, 1]
                imag_[0, i, j, 2] = img[i, j, 2]
        ####### PREDICTION
        pred_ = model_mask.predict(imag_[0:1, :, :, :], verbose=1)
        if pred_[0][0] > pred_[0][1]:
            print("With_mask")
            cv2.putText(
                face_img, "WITH MASK",
                (x,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,cv2.LINE_AA)
            cv2.rectangle(face_img, (x, y), (x + w, y + h), (0,255,0), 10)

        else:
            print("Without_mask")
            cv2.putText(
                face_img, "WITHOUT MASK",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,0,255),10)

    return face_img

while True:
    ret,frame=cap.read(0)
    frame=detect_face(frame)
    print(frame.shape)

    cv2.imshow('video',frame)

    if cv2.waitKey(1) & 0xFF==27:
        break

cv2.destroyAllWindows()
