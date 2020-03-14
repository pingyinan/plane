# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:33:36 2019

@author: PYN
"""
import cv2
import numpy as np
import math
from settings import Settings
import glob
import threading
import datetime
import time
import os

def find_red_region(parameters,hsv):
    red_mask1=cv2.inRange(hsv,parameters.red_lower1,parameters.red_upper1)
    red_mask2=cv2.inRange(hsv,parameters.red_lower2,parameters.red_upper2)
    red_mask=cv2.add(red_mask1,red_mask2)
    cv2.imshow('redarea', red_mask)
    cnts,hie=cv2.findContours(red_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def find_head_direction(parameters,hsv):
    try:
        blue_region=find_blue_region(parameters,hsv)
        (x1,y1),radius1 = cv2.minEnclosingCircle(blue_region[0])
        (x2,y2),radius2 = cv2.minEnclosingCircle(blue_region[1])
        if radius1>radius2:
            direction = (int(x2-x1),int(y2-y1))#x2-x1为负，机头相对于正方向顺时针偏了，正则逆时针偏
        else:
            direction = (int(x1-x2),int(y1-y2))
        return direction
    except:
        return None



#高空利用形态学寻找目标 and 红色区域找圆, 注意标志周围背景不能是白色，否则会形成连通区域干扰检测
def findtarget_method_5(parameters,img):
    Target = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)
    gray = cv2.medianBlur(gray, 5)
    ret, binary = cv2.threshold(gray, average*1.3, 255, cv2.THRESH_BINARY)#主要调整thresh
    binary = cv2.erode(binary, None, iterations=2)  # 腐蚀
    binary = cv2.dilate(binary, None, iterations=4)  # 膨胀
    # cv2.imshow('binary',binary)
    cnts0, hie = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for j in range(len(cnts0)):
        cnt = cnts0[j]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        cntArea = cv2.contourArea(cnt)
        if cntArea > 800:
            rect = cv2.minAreaRect(cnt)
            rectArea = rect[1][0] * rect[1][1]
            # print('rect', rect)
            # print('area bi:', cntArea / rectArea)
            if cntArea / rectArea > 0.85:#可调整是否是一个完整的矩形0.85
                if x-radius < 0:
                    xcmin = 0
                else:
                    xcmin = int(x - radius)
                if x+radius > parameters.photo_width:
                    xcmax = parameters.photo_width
                else:
                    xcmax = int(x + radius)
                if y-radius < 0:
                    ycmin = 0
                else:
                    ycmin = int(y - radius)
                if y+radius > parameters.photo_high:
                    ycmax = parameters.photo_high
                else:
                    ycmax = int(y + radius)
                #先进行灰度图找轮廓，一旦找到直接返回找到目标
                box = cv2.boxPoints(rect)
                if (parameters.method_5_flag == 0 or parameters.method_5_flag == 1):
                    region = gray[ycmin:ycmax, xcmin:xcmax]
                    # average = np.mean(region)
                    ret, binary = cv2.threshold(region, average * 1.3, 255, cv2.THRESH_BINARY_INV)
                    # cv2.imshow('huidu',binary)
                    cnts, hie = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    if len(cnts) > 0:
                        hi = hie[0]
                        for i in range(len(hi)):
                            if not (hi[i][2] == -1):
                                if hi[hi[i][2]][3] == i:
                                    (x0, y0), radius0 = cv2.minEnclosingCircle(cnts[i])
                                    (x1, y1), radius1 = cv2.minEnclosingCircle(cnts[hi[i][2]])
                                    nom = np.linalg.norm([int(x1 - x0), int(y1 - y0)], ord=2)
                                    bi = radius0 / radius1
                                    # print('nom', nom, 'bi', bi)
                                    if nom < 10 and bi < 1.8 and bi > 1.1 and radius0 < radius * 0.7:
                                        Target.append([round(x0, 2), round(y0, 2), round(radius0, 2), round(bi, 3)])
                        if len(Target) > 0: #灰度图法成功找到目标
                            parameters.method_5_flag = 1
                            i = np.argmax(np.mat(Target)[:, 2])
                            Target = Target[i]
                            return [True, (round(Target[0] + xcmin, 2), round(Target[1] + ycmin, 2), box),round(Target[2], 2), 1]  # r若能找到结构则找结构
                    #没有return说明没有找到目标
                    parameters.method_5_flag = 0
                if (parameters.method_5_flag == 0 or parameters.method_5_flag ==2):#红色掩模法
                    region = img[ycmin:ycmax,xcmin:xcmax]
                    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    red_mask1 = cv2.inRange(hsv, parameters.red_lower1, parameters.red_upper1)
                    red_mask2 = cv2.inRange(hsv, parameters.red_lower2, parameters.red_upper2)
                    red_mask3 = cv2.add(red_mask1, red_mask2)
                    red_mask = [red_mask1,red_mask2,red_mask3]
                    for mask in red_mask:
                        # cv2.imshow('yanmo', mask)
                        cnts_r, hie = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #注意这里红色掩模只取red_mask2
                        if len(cnts_r)>0:
                            c = max (cnts_r, key = cv2.contourArea)
                            (x2, y2), radius2 = cv2.minEnclosingCircle(c)
                            redArea = radius2 * radius2 * math.pi
                            # print('area bi2', redArea / rectArea)
                            if redArea/rectArea > 0.2:  #红色区域面积达到一定要求
                                # print('area bi2',redArea/rectArea)
                                nom = np.linalg.norm([int(x2+xcmin - x), int(y2+ycmin - y)], ord=2)
                                # print("norm",nom)
                                if nom < 5:  #红色区域与白色区域圆心重合
                                    parameters.method_5_flag = 2
                                    return [True, (round(x2 + xcmin, 2), round(y2 + ycmin, 2), box), round(radius2, 2),1]  # 找不到结构返回符合颜色条件的目标
                        #没有return则没有找到目标
                    parameters.method_5_flag = 0
    return None

def test():
    images=glob.glob('./get_picture/*.png')
    n = 1
    parameters = Settings()
    for image in images:
        img = cv2.imread(image)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(parameters.mtx, parameters.dist, (w, h), 1, (w, h))
        img = cv2.undistort(img, parameters.mtx, parameters.dist, None, newcameramtx)
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]
        parameters.photo_width = w
        parameters.photo_high = h
        message = findtarget_method_5(parameters, img)
        if not (message is None):
            cv2.circle(img, (int(message[1][0]),int(message[1][1])), int(message[2]), [255, 0, 0], 1)
        cv2.imwrite('./picture1/%d'%(n) + '.png',img)
        n = n+1
def findtarget_inpictures_5():
    x = 0
    for root, dirs, files in os.walk('./shiyantupian'):
        for d in dirs:
            print(d)
        for file in files:
            print(file)
            img_path = root + '/' + file
            img = cv2.imread(img_path)
            # print(img_path, img.shape)
            #图像处理
            parameters = Settings()
            img = correctdistortion(parameters, img)
            img = whitebalance(img)
            message = findtarget_method_5(parameters, img)
            if not (message is None):
                cv2.circle(img, (int(message[1][0]), int(message[1][1])), int(message[2]), [255, 0, 0], 1)
                cv2.line(img, (int(message[1][2][0][0]), int(message[1][2][0][1])),
                         (int(message[1][2][1][0]), int(message[1][2][1][1])), (0, 0, 255), 3)
                cv2.line(img, (int(message[1][2][1][0]), int(message[1][2][1][1])),
                         (int(message[1][2][2][0]), int(message[1][2][2][1])), (0, 0, 255), 3)
                cv2.line(img, (int(message[1][2][2][0]), int(message[1][2][2][1])),
                         (int(message[1][2][3][0]), int(message[1][2][3][1])), (0, 0, 255), 3)
                cv2.line(img, (int(message[1][2][3][0]), int(message[1][2][3][1])),
                         (int(message[1][2][0][0]), int(message[1][2][0][1])), (0, 0, 255), 3)
            x = x + 1
            img_saving_path = './result/' + str(x) + '.png'
            print(img_saving_path)
            cv2.imwrite(img_saving_path, img)

def test_findtarget_method_5():
    img = cv2.imread('./shiyantupian/2/81.png')
    cv2.imshow('yuan tu',img)
    parameters = Settings()
    s1 = time.clock()
    img = correctdistortion(parameters, img)
    img = whitebalance(img)

    message = findtarget_method_5(parameters, img)
    s2 = time.clock()
    print('time',s2 - s1)

    if not (message is None):
        print('find Target', message)
        cv2.circle(img, (int(message[1][0]),int(message[1][1])), int(message[2]), [255,0 , 0], 1)
        cv2.line(img, (int(message[1][2][0][0]), int(message[1][2][0][1])), (int(message[1][2][1][0]), int(message[1][2][1][1])), (0, 0, 255), 3)
        cv2.line(img, (int(message[1][2][1][0]), int(message[1][2][1][1])), (int(message[1][2][2][0]), int(message[1][2][2][1])), (0, 0, 255), 3)
        cv2.line(img, (int(message[1][2][2][0]), int(message[1][2][2][1])), (int(message[1][2][3][0]), int(message[1][2][3][1])), (0, 0, 255), 3)
        cv2.line(img, (int(message[1][2][3][0]), int(message[1][2][3][1])), (int(message[1][2][0][0]), int(message[1][2][0][1])), (0, 0, 255), 3)
    else:
        print('target is not found')
    cv2.imshow('img',img)
    cv2.waitKey(0)

#红白标志，低空检测嵌套圆环
#返回值：无目标——None
#     ：有目标—— [bool, (x,y), radius, circlenumber]
def findtarget_method_6(parameters,frame):
    Target = []
    # 红色环，红色掩模
    if (parameters.method_6_flag == 0 or parameters.method_6_flag == 1):
        img = frame.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, parameters.red_lower1, parameters.red_upper1)
        red_mask2 = cv2.inRange(hsv, parameters.red_lower2, parameters.red_upper2)
        red_mask = cv2.add(red_mask1, red_mask2)
        # cv2.imshow('1',red_mask)
        red_mask = cv2.erode(red_mask, None, iterations=2)  # 腐蚀
        red_mask = cv2.dilate(red_mask, None, iterations=2)  # 膨胀
        # cv2.imshow('red', red_mask)
        cnts, hie = cv2.findContours(red_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            hi = hie[0]
            for i in range(len(hi)): #排除面积小的轮廓
                if cv2.contourArea(cnts[i]) < 400:
                    continue
                if not (hi[i][2] == -1):
                    if hi[hi[i][2]][3] == i:
                        (x0, y0), radius0 = cv2.minEnclosingCircle(cnts[i])
                        # cv2.circle(frame,(int(x0),int(y0)),int(radius0),(0, 0, 255),2)
                        (x1, y1), radius1 = cv2.minEnclosingCircle(cnts[hi[i][2]])
                        # cv2.circle(frame, (int(x1), int(y1)), int(radius1), (0, 255, 0), 2)
                        nom = np.linalg.norm([int(x1 - x0), int(y1 - y0)], ord=2)
                        bi = radius0 / radius1
                        # print('nom',nom,'bi',bi)
                        if (bi < 1.8 and bi > 1.1) and (nom < 10) or (nom < 1) or (bi < 1.55 and bi > 1.45) or (bi < 1.3 and bi > 1.2):  # 判断条件可调整
                            Target.append([round(x0, 2), round(y0, 2), round(radius0, 2), round(bi, 3)])
        if len(Target) > 0:
            parameters.method_6_flag = 1
        else:
            parameters.method_6_flag = 0
    if (parameters.method_6_flag == 0 or parameters.method_6_flag == 2):
        # #白色环，灰度图像二值化
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        average = np.mean(gray)
        gray = cv2.medianBlur(gray, 7)
        threshold = 1.4 * average
        ret, binary = cv2.threshold(gray, threshold, 255,cv2.THRESH_BINARY_INV)  # 主要调节threshold,背景暗,标志区别度明显则调高,背景亮，标志区别度不明显则调低,注意threshold不能超过255
        binary = cv2.erode(binary, None, iterations=1)  # 腐蚀
        binary = cv2.dilate(binary, None, iterations=1)  # 膨胀
        cv2.imshow('binary', binary)
        cnts, hie = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            hi = hie[0]
            for i in range(len(hi)):
                if not (hi[i][2] == -1):
                    if hi[hi[i][2]][3] == i:
                        (x0, y0), radius0 = cv2.minEnclosingCircle(cnts[i])
                        # cv2.circle(img, (int(x0), int(y0)), int(radius0), (0, 0, 255), 2)
                        (x1, y1), radius1 = cv2.minEnclosingCircle(cnts[hi[i][2]])
                        # cv2.circle(img, (int(x1), int(y1)), int(radius1), (0, 255, 0), 2)
                        nom = np.linalg.norm([int(x1 - x0), int(y1 - y0)], ord=2)
                        bi = radius0 / radius1
                        # print('nom', nom, 'bi', bi)
                        if (bi < 1.8 and bi > 1.1) and (nom < 10) or (nom < 1) or (bi < 1.55 and bi > 1.45) or (bi < 1.3 and bi > 1.2):  # 判断条件可调整
                            Target.append([round(x0, 2), round(y0, 2), round(radius0, 2), round(bi, 3)])
        if len(Target) > 0:
            parameters.method_6_flag = 2
        else:
            parameters.method_6_flag = 0
    number = len(Target)
    if number > 0: #检测到目标的前提下，返回最大圆的信息，现需要判断检测到的最大圆是第几个圆
        i = np.argmax(np.mat(Target)[:, 2])
        Target = Target[i]
        if parameters.lastRadius == 0:
            parameters.lastRadius = Target[2]
            parameters.lastbi = Target[3]
        else:
            relativechange = (Target[2] - parameters.lastRadius) / parameters.lastRadius
            if abs(relativechange) > 0.2: # 圆环大小是否有突变,进一步判断是否为正常突变，还是异常突变
                if parameters.ifchange[0] == 0:
                    parameters.ifchange[0] = 1
                    parameters.ifchange[1] = 1
                else:
                    parameters.ifchange[1] = parameters.ifchange[1] + 1
                    if parameters.ifchange[1] > 10:#连续10次检测到圆环改变,则真的改变，判断怎么改变
                        if relativechange < 0:#圆减小
                            if parameters.lastbi > 1.375 and Target[3] < 1.375: #前一个检测圆环bi = 1.5 , 当前圆环bi = 1.25,说明当前检测到第2个圆环
                                parameters.lastRadius = Target[2]
                                parameters.lastbi = Target[3]
                                parameters.circlenumber = 2
                                parameters.ifchange = [0, 0]
                                message = [True, (round(Target[0], 2), round(Target[1], 2)), round(Target[2], 2), parameters.circlenumber]
                                return message
                            elif parameters.lastbi < 1.375 and Target[3] > 1.375: #前一个检测圆环bi = 1.25 , 当前圆环bi = 1.5,说明当前检测到第3个圆环
                                parameters.lastRadius = Target[2]
                                parameters.lastbi = Target[3]
                                parameters.circlenumber = 3
                                parameters.ifchange = [0, 0]
                                message = [True, (round(Target[0], 2), round(Target[1], 2)), round(Target[2], 2), parameters.circlenumber]
                                return message
                        if relativechange > 0:
                            if parameters.lastbi > 1.375 and Target[3] < 1.375: #前一个3圆环，现在检测到2圆环
                                parameters.lastRadius = Target[2]
                                parameters.lastbi = Target[3]
                                parameters.circlenumber = 2
                                parameters.ifchange = [0, 0]
                                message = [True, (round(Target[0], 2), round(Target[1], 2)), round(Target[2], 2), parameters.circlenumber]
                                return message
                            if parameters.lastbi < 1.375 and Target[3] > 1.375: #前一个2圆环，现在检测到1圆环
                                parameters.lastRadius = Target[2]
                                parameters.lastbi = Target[3]
                                parameters.circlenumber = 1
                                parameters.ifchange = [0, 0]
                                message = [True, (round(Target[0], 2), round(Target[1], 2)), round(Target[2], 2), parameters.circlenumber]
                                return message
                #检测到突变，但是没有达到改变条件，依旧输出原来的半径和圆环编号
                message = [True, (round(Target[0], 2), round(Target[1], 2)), parameters.lastRadius, parameters.circlenumber]
                # parameters.abnormalData == parameters.abnormalData.append(Target.insert(1, 'error_circleNumber'))
                return message
            else:#没有突变，更新
                parameters.ifchange = [0,0]
                parameters.lastRadius = Target[2]
                parameters.lastbi = Target[3]
        message = [True, (round(Target[0], 2), round(Target[1], 2)), round(Target[2], 2), parameters.circlenumber] # 没有突变直接输出
        return message
    return None

def test_findtarget_method_6():
    img = cv2.imread('./shiyantupian/3/49.png')

    # img = cv2.imread('./picture1/15.jpg')
    parameters = Settings()
    #畸变矫正
    s2 = time.clock()
    img = correctdistortion(parameters, img)
    # 白平衡
    s3 = time.clock()
    img = whitebalance(img)
    s4 = time.clock()
    # cv2.imshow('yuantu', img)
    message = findtarget_method_6(parameters, img)
    s5 = time.clock()
    print('time1',s3 - s2, 'time2', s4 - s3, 'time3', s5 - s4, parameters.method_6_flag)
    if not (message is None):
        print(message)
        cv2.circle(img, (int(message[1][0]),int(message[1][1])), int(message[2]), [255, 0, 0], 2)
    else:
        print('target is not found')
    cv2.imshow('img',img)
    cv2.waitKey(0)

#最低检测标准，只检测红色圆形目标
def findtarget_method_7(parameters, img):
    Target = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, parameters.red_lower1, parameters.red_upper1)
    red_mask2 = cv2.inRange(hsv, parameters.red_lower2, parameters.red_upper2)
    red_mask = cv2.add(red_mask1, red_mask2)
    cv2.imshow('red_mask',red_mask)
    cnts, hie = cv2.findContours(red_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)>0:
        for cnt in cnts:
            Area = cv2.contourArea(cnt)
            if Area < 400:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if Area / (math.pi * radius * radius) > 0.9:
                Target.append([round(x, 2), round(y, 2), round(radius, 2) ])
        if len(Target) > 0:
            i = np.argmax(np.mat(Target)[:, 2])
            Target = Target[i]
            return [True, (round(Target[0], 2), round(Target[1], 2)), round(Target[2],2), 4]
    return None

def test_findtarget_method_7():
    img = cv2.imread('./shiyantupian/1/23.png')
    cv2.imshow('yuantu',img)
    # img = cv2.imread('./picture1/15.jpg')
    parameters = Settings()
    #畸变矫正
    s2 = time.clock()
    img = correctdistortion(parameters, img)
    # 白平衡
    s3 = time.clock()
    img = whitebalance(img)
    s4 = time.clock()
    message = findtarget_method_7(parameters, img)
    s5 = time.clock()
    print('time1',s3 - s2, 'time2', s4 - s3, 'time3', s5 - s4)
    if not (message is None):
        print(message)
        cv2.circle(img, (int(message[1][0]),int(message[1][1])), int(message[2]), [255, 0, 0], 2)
        cv2.circle(img, (int(message[1][0]), int(message[1][1])), 20, [255, 0, 0], 2)
    else:
        print('target is not found')
    cv2.imshow('img',img)
    cv2.waitKey(0)


def findTarget(parameters, img):
    if parameters.flag == 2:#第一次判断用什么方法
        message1 = findtarget_method_5(parameters, img)
        message2 = findtarget_method_6(parameters, img)
        if not message1 is None:
            if not message2 is None:
                if message2[2] > message1[2]:
                    parameters.flag = 1
                else:
                    parameters.flag = 0
            else:
                parameters.flag = 0
        elif not message2 is None:
            parameters.flag = 1
        return [False, None, None, None, ]
    if parameters.flag == 0:
        message = findtarget_method_5(parameters, img)  # 10-3米
        if message == None:
            message =findtarget_method_6(parameters, img)
            if not (message is None):
                message[3] = 1
                if message[2] > 60:
                    parameters.flag = 1
    elif parameters.flag == 1:
        message = findtarget_method_6(parameters, img)
        if message is None:
            if parameters.circlenumber == 3:
                message = findtarget_method_7(parameters, img)
            elif parameters.circlenumber == 1:
                message = findtarget_method_5(parameters, img)
    if message is None:
        message = [False, None, None, None, ]
    return message

def findTarget2(parameters, img):
    if parameters.flag == 0: # 10-3米下的检测策略
        message = findtarget_method_5(parameters, img)
        #凡是检测到目标，都将圆环标号置1
        if message is not None:
            message[3] = 1
            if message[2] > 60: #当检测到的圆环半径达到60个像素时，高度达到3m
                parameters.flag = 1
    elif parameters.flag == 1: # <3m的检测策略
        message = findtarget_method_6(parameters, img)
        if message is None:
            if parameters.circlenumber == 3:
                message = findtarget_method_7(parameters, img)
#判断是否找到目标
    if message is None :
        message = [False, None ,None, None]
        if parameters.isfindTarget[0] == True: #目标丢失
            parameters.isfindTarget[1] = parameters.isfindTarget[1] + 1
            if parameters.isfindTarget[1] == 5: #连续5次没有检测到目标则视为目标丢失
                parameters.isfindTarget[0] = False
                parameters.isfindTarget[1] = 0
        if parameters.isfindTarget[0] == False: #10次里有一次没找到目标则重新基数
            parameters.isfindTarget[1] = 0
    if message is not None :
        if parameters.isfindTarget[0] == False :
            parameters.isfindTarget[1] = parameters.isfindTarget[1] + 1
            if parameters.isfindTarget[1] == 10: #连续10次找到目标才算找到目标
                parameters.isfindTarget[0] = True
                parameters.isfindTarget[1] = 0
        if parameters.isfindTarget[0] == True: #5次里有一次没找到目标则重新基数
            parameters.isfindTarget[1] = 0
    #返回信息
    if parameters.isfindTarget[0] == False:
        return [False, None, None, None ]
    else:
        return message


def test_findtarget():
    parameters = Settings()
    p = 0
    cap = cv2.VideoCapture(p)
    while not cap.isOpened():
        p = p + 1
        print('camera is not opened %d' % p)
        cap = cv2.VideoCapture(p)
    time.sleep(1)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        img = frame.copy()

        if ret == 1:
            message = findTarget2(parameters, img)
            if message is not None and message[0] is True:
                print(message)
                if message[2] is None :
                    cv2.circle(img, (int(message[1][0]), int(message[1][1])), 4, [0, 0, 255], 3)
                else:
                    cv2.circle(img, (int(message[1][0]), int(message[1][1])), int(message[2]), [255, 0, 0], 3)
                    cv2.circle(img, (int(message[1][0]), int(message[1][1])), 4, [0, 0, 255], 3)
            else:
                print('target is not found')
            cv2.putText(img, "methond:%d ,circlenumber:%d" %(parameters.flag, parameters.circlenumber), (img.shape[1] - 620, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 1)
            cv2.imshow('img', img)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break


def testcamera():
    p = 0
    cap = cv2.VideoCapture(p)
    while not cap.isOpened():
        p = p + 1
        print('camera is not opened %d' % p)
        cap = cv2.VideoCapture(p)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        if ret == 1:
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break




# 打开摄像头找目标，可以保存视频
def findTargetInCap():
    parameters = Settings()
    p = 0
    cap = cv2.VideoCapture(0)
    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 8, (640, 480))
    while not cap.isOpened():
        p = p + 1
        print('camera is not opened ', p)
        cap = cv2.VideoCapture(p)
    cap.set(3, parameters.photo_width)
    cap.set(4, parameters.photo_high)
    cap.set(5, 8)
    while True:
        ret, img = cap.read()
        if ret == 1:
            # h, w = img.shape[:2]
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(parameters.mtx, parameters.dist, (w, h), 0, (w, h))
            # dst = cv2.undistort(img, parameters.mtx, parameters.dist, None, newcameramtx)
            # GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # MD = cv2.medianBlur(GrayImage, 5)

            Target = findtarget_method_1(parameters, img)
            if not (Target is None):
                h = distance_to_camera(parameters, Target[2])
                cv2.putText(img, "%.2fcm" % (h * 100), (img.shape[1] - 400, img.shape[0] - 20),cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
                cv2.circle(img, (int(Target[0]), int(Target[1])), int(Target[2]), (0, 0, 255), 2)
                print('find target')
            else:
                print('target is not found')
            cv2.imshow('img', img)
            # out.write(img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

def processpicture():
    parameters = Settings()
    p = 0
    cap = cv2.VideoCapture(p)
    while not cap.isOpened():
        p = p + 1
        print('camera is not opened %d' % p)
        cap = cv2.VideoCapture(p)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(5, 8)
    while True:
        img = cap.read()
        #畸变矫正
        img = correctdistortion(parameters, img)
        #白平衡
        img = whitebalance(img)

def correctdistortion(parameters, img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(parameters.mtx, parameters.dist, (w, h), 1, (w, h))
    img = cv2.undistort(img, parameters.mtx, parameters.dist, None, newcameramtx)
    x, y, w, h = roi
    img = img[y:y + h, x:x + w]
    parameters.photo_width = w
    parameters.photo_high = h
    return img
#白平衡会使得白色区域检测为红色redmask1,白平衡对灰度图影响不大，但对颜色检测影响很大
def whitebalance(img):

    dst = cv2.resize(img, (32,24), 0, cv2.INTER_AREA)
    b, g, r = cv2.split(dst)
    bmean = np.mean(b)
    gmean = np.mean(g)
    rmean = np.mean(r)
    b, g, r = cv2.split(img)

    kb = (bmean + gmean + rmean) / (3 * bmean)
    kg = (bmean + gmean + rmean) / (3 * gmean)
    kr = (bmean + gmean + rmean) / (3 * rmean)

    b = b * kb
    if kb > 1:
        b[b > 255] = 255
    b = b.astype(np.uint8)

    g = g * kg
    if kg > 1:
        g[g > 255] = 255
    g = g.astype(np.uint8)

    r = r * kr
    if kr > 1:
        r[r > 255] = 255
    r = r.astype(np.uint8)
    img_0 = cv2.merge([b, g, r])
    return img_0
