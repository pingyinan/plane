import cv2
import numpy as np

class Settings():
    def __init__(self):
        self.photo_width=640
        self.photo_high=480
        self.flag=0
        #红色圆环外圆，单位cm
        self.circle_1 = 30
        self.circle_2 = 10
        self.circle_3 = 6
        self.circle_4 = 2
        self.red_lower1=np.array([0,43,46])
        self.red_upper1=np.array([10,255,255])
        self.red_lower2=np.array([156,43,46])
        self.red_upper2=np.array([180,255,255])
        self.focalLength=603
        self.mtx=np.array([[604.719,0,309.609],[0,602.588,238.048],[0,0,1]])
        self.dist=np.array([-0.464793,0.259029,0.001081,0.001159])
        self.lastRadius = 0
        self.lastbi = 0
        self.circlenumber = 1
        self.abnormalData = []
        self.ifchange = [0,0]
        self.method_6_flag = 0 #0两种方法都可以，1红色掩模法，2灰度图法
        self.method_5_flag = 0 #0两种方法都可以，1灰度图法，2红色掩模法
        self.isfindTarget = [False, 0] #判断是否找到目标