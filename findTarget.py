import cv2
from settings import Settings
import numpy as np
from functions import findtarget_method_2
parameters = Settings()
p = 0
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 8, (640,480))
while not cap.isOpened():
    p = p + 1
    print('camera is not opened ', p)
    cap = cv2.VideoCapture(p)
cap.set(3, parameters.photo_width)
cap.set(4, parameters.photo_high)
cap.set(5,8)
while True:
    ret, img = cap.read()
    if ret == 1:
        # h, w = img.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(parameters.mtx, parameters.dist, (w, h), 0, (w, h))
        # dst = cv2.undistort(img, parameters.mtx, parameters.dist, None, newcameramtx)
        # GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # MD = cv2.medianBlur(GrayImage, 5)
        try:
            x, y, R = findtarget_method_2(parameters, img)
            cv2.circle(img,(x,y),R,(255,0,0),2)
            print('find target')
        except:
            print('target is not found')
        cv2.imshow('img',img)
        out.write(img)
# cv2.imshow('GrayImage', MD)
# th2 = cv2.adaptiveThreshold(MD, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
# cv2.imshow('th2', th2)
# imgray = cv2.Canny(MD, 30, 400)
# cv2.imshow('imgray', imgray)
# circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=20, minRadius=2, maxRadius=80)
        if cv2.waitKey(1)==27:
            cv2.destroyAllWindows()
            break