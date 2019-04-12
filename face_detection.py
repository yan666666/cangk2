#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import cv2
import time
import rospy
from std_msgs.msg import String


class find_people():
    def __init__(self):
        rospy.init_node("find_people")
        self.pub = rospy.Publisher('/xfwords', String, queue_size=15)
        self.detector = dlib.get_frontal_face_detector()
        self.people_num = 0
        self.keyboard_control()

    def read_capture(self):
        # open the video camara
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()

        while success:
            img_np = frame.copy()

            # dlib的人脸检测器只能检测80x80和更大的人脸，如果需要检测比它小的人脸，需要对图像上采样，一次上采样图像尺寸放大一倍
            # rects = detector(img,1) #1次上采样
            rects = self.detector(img_np, 0)
            print "people_num : ", self.people_num
            if len(rects) != 0:
                number = len(rects)
                if self.people_num != number:
                    self.people_num = number
                    if number == 1:
                        result = "一"
                    elif number == 2:
                        result = "两"
                    elif number == 3:
                        result = "三"
                    elif number == 4:
                        result = "四"
                    elif number == 5:
                        result = "五"
                    self.pub.publish("我看到 " + result + " 个人")


            cv2.imshow('capture face detection', img_np)

            if cv2.waitKey(1) >= 0:
                break
            success, frame = cap.read()
        cv2.destroyAllWindows()
        cap.release()

    def keyboard_control(self):
        command = ''
        while command != 'c':
            try:
                command = raw_input('next command : ')
                if command == 'r':
                    self.read_capture()
                else:
                    print("Invalid Command!")
            except Exception as e:
                print e


if __name__ == '__main__':
    find_people()