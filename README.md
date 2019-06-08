# cangk2
    我们组准备基于Robocup2019规则中Restaurant部分做一个智能酒保机器人，完成在起始地点接收语音指令，前往目标地点，用语音传达出顾客的需要并且等待机械臂成功抓取物品,然后自主导航回到起始地点，将物品递给客户。
创新点：机器人利用语音主动询问顾客需求，并与顾客交流，满足顾客需要。
本人负责大作业机械臂部分，根据2019年比赛规则的handover和Take and serve an order部分打算做一个与导航结合的抓取物品，传递物品，放置到指定位置的机械臂。   智能大会作业  
                                                         浅谈松下焊接机器人  
    近些年的智能大会上，松下每次都是带着最新一代的焊接机器人参加，我对此的印象十分深刻，于是从技术角度对此现象的原因进行了简单分析。
松下的焊接机器人一般有机器人本体、全数字焊机、焊枪、机器人示教器、机器人控制枪等组成。有手动编程、离线编程等多种模式，配备全面的可视化界面，使商品有很强的竞争力。  
    值得一提的是松下的激光系统，是世界上唯一的为客户提供一体化解决方案的高级系统。其中具有代表性的就是LAOPRISS激光器。采用光束合成技术（WBXC）研发出了高亮度直接半导体激光器，将多个低功率的激光束通过波长合成技术耦合成一系列高功率激光束，满足激光远程焊接和切割的要求，称为直接半导体激光器（DDL）。相比于其他激光器，DDL有更好的效率，可靠性，更小的尺寸和更低的成本。  
    还有就是精度要求了。因为激光头产生的光斑直径一般为零点几毫米，且为扫面头，可以实现光斑一边旋转一边在机器人的带动下直线运动。这种方式称为螺旋焊接，可以大大降低焊接缺陷。  
    最后一件我觉得松下可以一直在智能大会上展示此类机器人的原因就是其简单的操作方式了。运用数字通讯和激光焊接导航系统，控制器内接专家库（用到了专家系统的机器学习相关知识）可以简单的设置材质、板厚、接头形式等关键要求就可以完成复杂的焊接工作。  
    总之，连续几年松下焊接机器人在智能大会上的出现确实让我感受到了这些相关技术的重要性与实用性。期待下次智能大会上松下焊接机器人新功能的出现！  

##Final Summary Report Of Robot Software Engineering  
Name：Zhang Yan  
Student Number：1611466  
Date:June,2019  
###Contents  
I  Project Introduction  
Ⅱ  Basic Function Analysis  
Ⅲ  Technical Route  
IV  Specific Realization Method  
V  Project Completion  
VI  Summary and Reflection  
###I  Project Introduction  
Our group has built an intelligent bartender robot based on the Restaurant part of Robocup 2019 rule. It receives voice instructions at the starting place, goes to the target place, conveys customers' needs by voice and waits for the robot arm to grab the object successfully. Then it navigates back to the starting place and delivers the object to the customers.  
The robot use voice to actively inquire about customer needs and communicate with customers to meet customer needs.I am responsible for the arm part. According to the handover，take and serve an order parts of the rules of the competition in 2019, I intend to make a grab objects combined with navigation, transfer objects and place them in a designated position.  
###Ⅱ  Basic Function Analysis  
Object recognition, fine-tuning of fixed-point position and joints control of the arm are used to grab objects.
Among them, object recognition can be solved by darknet_ros, and robot position fine-tuning can be realized by Kinect image information. The robot can grab objects under a more precise condition only after it reaches the designated position through fixed-point navigation. The actual process is like this： Drinks are placed on the bar's specific counter, so it is necessary for the robot to navigate to the location of the counter and grab the object. We can use the navigation package to realize Fixed-point navigation.It identify the objects to be grabbed after reaching the approximate position and adjust its position according to the location of the objects, so that the robot is just in front of the objects to be grabbed. That is convenient for grasping.  
As for the joint control of the arm, it can be realized by calculating the angle of each joint when grasping the object through Inverse Kinematics. The robot has arrived at the front of the object. If we give the angle information of each joint directly as arm-dance.py, which our teacher said in the previous lesson, we can complete the grabbing of the object.  
After arriving at the customer's location,the robot needs to adjust the posture of the arm to place the object on the table or pass it to the customer.This can also be achieved by fixing the joint angle of the arm.  
###Ⅲ  Technical Route  
The first problem to be solved is to minimize the grabbing error. There are two main ways to reduce the grabbing error. One is to use the base of the robot to optimize the navigation, so that the position of the robot parking is more accurate and it is convenient for the robot to grab the object directly. The second method is to minimize the error of the arm as much as possible and to use more precise components. Obviously, the second method is not feasible. Because the precision of the arm used in the experiment is very accurate,.But its grabbing range is small, so it is not easy to grasp things. So we choose the first method, using navigation to reduce errors.
Here is the code for tuning.The whole document is divided into four main parts. The first part is using Darknet_ros to identify specific objects, and using BoundingBox to frame the range of objects in the image. In the code, I write "bottle" to replace the object. The second part calculates the average distance between the point cloud and the camera by using the depth image information of Kinect and the topic name is "/camera/depth/image_raw". The third part is to use Twist to publish the angle control information to the robot by comparing the position of the identified object in the image with the middle of the screen. If the deviation is large, a larger angular velocity will be published. And if the deviation is small, a smaller angular velocity will be published. When the difference is within the tolerance range, the angle is considered to have been adjusted well. The fourth part compares the average depth of the object in the image with the robot. "Twist" is used to distribute linear speed control information to the robot. If the deviation is large, a larger linear velocity will be published, and if the deviation is small, a smaller linear velocity will be published. When the difference is within the tolerance range, the distance is considered to have been adjusted well.  
        #! /usr/bin/env python
        # -*- coding: utf-8 -*-
        """
            Date: 2019/06/06
            Author: yan666
            Abstract: from darknet get bounding box messages and depth image to find the coordinates of bottle
        """
        import roslib
        import rospy
        from std_msgs.msg import String
        import os
        import cv2
        import sys
        import time
        import numpy as np
        from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
        from geometry_msgs.msg import Twist

        from cv_bridge import CvBridge, CvBridgeError
        from sensor_msgs.msg import Image

        class catch_object(object):

            def __init__(self):
                rospy.on_shutdown(self.cleanup)

                self.pub_twist_topic_name           = None
                self.sub_depth_image_topic_name     = None
                self.sub_is_reach_topic_name        = None
                self.sub_detect_result_topic_name   = None
                self.twist_msg = Twist()
                self.depth_img = np.array((480,640,2))
                self.avg_depth = 0.0

                # Variables
                self.is_reach = False
                self.is_get_image = False
                self.is_find_bottle = False
                self.is_turn_angular = False
                self.is_turn_linear = False
                self.is_pub_nav2arm = False

                self.get_params()
                print('[INFO] Start Turning Distance and Angular')

            def get_params(self):
                self.sub_depth_image_topic_name     = rospy.get_param("sub_depth_image_topic_name", "/camera/depth/image_raw")
                self.sub_is_reach_topic_name        = rospy.get_param("sub_is_reach_topic_name", "/catch_start")
                self.sub_detect_result_topic_name   = rospy.get_param('sub_detect_result_topic_name', '/darknet_ros/bounding_boxes')
                self.pub_twist_topic_name           = rospy.get_param('pub_twist_topic_name', '/cmd_vel_mux/input/navi')
                self.pub_nav2arm_topic_name         = rospy.get_param('pub_nav2arm_topic_name', '/nav2arm')

                self.pub_nav2arm = rospy.Publisher(self.pub_nav2arm_topic_name, String, queue_size=1)
                self.pub_twist = rospy.Publisher(self.pub_twist_topic_name, Twist, queue_size=1)
                rospy.Subscriber(self.sub_is_reach_topic_name, String, self.isReachCallback)
                rospy.Subscriber(self.sub_depth_image_topic_name, Image, self.depthimgCallback)
                rospy.Subscriber(self.sub_detect_result_topic_name, BoundingBoxes, self.resultCallback)

            def isReachCallback(self, msg):
                if msg.data == 'ready_to_catch':
                    self.is_reach = True

            def depthimgCallback(self, msg):
                if self.is_reach:
                    bridge = CvBridge()
                    self.depth_img = bridge.imgmsg_to_cv2(msg, msg.encoding)
                    cv2.imshow('img', self.depth_img)
                    cv2.waitKey(30)
                    self.is_get_image = True
                else:
                    #print('The robot has not reached the destination')
                    pass

            def resultCallback(self, msg):
                #if self.is_turn_angular and self.is_turn_linear and self.is_pub_nav2arm == False:
                if self.is_turn_linear and self.is_pub_nav2arm == False:
                    nav2arm_msg = String()
                    nav2arm_msg.data = 'catch'
                    self.pub_nav2arm.publish(nav2arm_msg)
                    self.is_pub_nav2arm = True

                if self.is_get_image:

                    boundingbox = BoundingBox()
                    boundingbox = msg.bounding_boxes
                    #wprint(boundingbox)
                    for bx in boundingbox:
                        if bx.Class == 'bottle':
                            xmin = bx.xmin
                            xmax = bx.xmax
                            ymin = bx.ymin
                            ymax = bx.ymax
                            middle_x = (xmin+xmax)/2
                            self.is_find_bottle = True

                    if self.is_find_bottle:
                        if self.is_turn_angular==False:
                            if middle_x-320>50:
                                self.twist_msg.angular.z = -0.3
                                self.pub_twist.publish(self.twist_msg)
                                self.is_get_image = False
                            elif middle_x-320>10:
                                self.twist_msg.angular.z = -0.1
                                self.pub_twist.publish(self.twist_msg)
                                self.is_get_image = False
                            elif middle_x-320<-50:
                                self.twist_msg.angular.z = 0.3
                                self.pub_twist.publish(self.twist_msg)
                                self.is_get_image = False
                            elif middle_x-320<-10:
                                self.twist_msg.angular.z = 0.1
                                self.pub_twist.publish(self.twist_msg)
                                self.is_get_image = False
                            else:
                                self.is_turn_angular = True
                                self.is_get_image = False
                                print('[INFO] The Angular is Correct')
                        if self.is_turn_angular and self.is_turn_linear==False:
                            self.avg_depth = 0.0
                            count = (xmax-xmin)*(ymax-ymin)
                            for i in range(int(xmin), int(xmax)):
                                for j in range(int(ymin), int(ymax)):
                                    tmp_depth = self.depth_img[i,j]
                                    print(tmp_depth)
                                    if tmp_depth > 1000:
                                        count = count-1
                                    else:
                                        self.avg_depth = self.avg_depth + tmp_depth
                            self.avg_depth = self.avg_depth/count
                            #print(self.avg_depth)
                            if self.avg_depth-1>500:
                                self.twist_msg.linear.x = 0.3
                                self.pub_twist.publish(self.twist_msg)
                                self.is_get_image = False
                            elif self.avg_depth-1>2:
                                self.twist_msg.linear.x = 0.1
                                self.pub_twist.publish(self.twist_msg)
                                self.is_get_image = False
                            elif self.avg_depth-1<-500:
                                self.twist_msg.linear.x = -0.3
                                self.pub_twist.publish(self.twist_msg)
                                self.is_get_image = False
                            elif self.avg_depth-1<-2:
                                self.twist_msg.linear.x = -0.1
                                self.pub_twist.publish(self.twist_msg)
                                self.is_get_image = False
                            else:
                                self.is_turn_linear = True
                                self.is_get_image = False
                                print('[INFO] The Distance is Correct')

                    else:
                        print('[INFO] Cannot Find Target Object')
                else:
                    print('[INFO] Cannot Get Valid Image')

            def cleanup(self):
                print('[INFO] I have finished the task')

        if __name__ == '__main__':
            rospy.init_node("catch_object", anonymous=True)
            ctrl = catch_object()
            rospy.spin()
The The second problem to be solved is the control of the arm.I wrote the code which includes the angle information of each joint of the arm to complete each specific position based on the example of arm_dance.py.It is the one that our teacher talked about in class. It is mainly divided into four postures. The first is the initial arm posture, to ensure that the work area is not obstructed. The second is the position of the arm when the robot arrives at the counter, which is already aimed at the object being grabbed. The third is the grasping posture, which requires lifting the object on the basis of the second posture and ensuring the integrity of the object. The fourth is the posture of placing objects, which should be placed at the height of the desktop and slowly released. In this way, as long as the corresponding position and posture instructions are issued according to the completion time of each task, the corresponding actions can be completed.Here is the code.  
    #!/usr/bin/env python

    """
        arm.py - move robot arm according to predefined gestures

    """

    import rospy
    from std_msgs.msg import Float64
    from std_msgs.msg import String
    import time

    class Loop:
        def __init__(self):
            rospy.on_shutdown(self.cleanup)

        # publish command message to joints/servos of arm
            self.joint1 = rospy.Publisher('/waist_controller/command',Float64)
        self.joint2 = rospy.Publisher('/shoulder_controller/command',Float64)
            self.joint3 = rospy.Publisher('/elbow_controller/command',Float64)
            self.joint4 = rospy.Publisher('/wrist_controller/command',Float64)
        self.joint5 = rospy.Publisher('/hand_controller/command',Float64)
        self.pos1 = Float64()
            self.pos2 = Float64()
            self.pos3 = Float64()
            self.pos4 = Float64()
            self.pos5 = Float64()

        # Initial gesture of robot arm
        self.pos1 = 0.0
        self.pos2 = -2.09
        self.pos3 = 2.4
        self.pos4 = 1.04
        self.pos5 = -0.4
        self.joint1.publish(self.pos1)
        self.joint2.publish(self.pos2)
        self.joint3.publish(self.pos3)
        self.joint4.publish(self.pos4)
        self.joint5.publish(self.pos5)

        pub = rospy.Publisher('arm2navi', String, queue_size=1)

        while not rospy.is_shutdown():
            time.sleep(2)
            # gesture 1
            self.pos1 = 0.0
            self.pos2 = -2.09
            self.pos3 = 2.61
            time.sleep(5)
            self.pos4 = 1.04
            self.pos5 = -0.4
            self.joint1.publish(self.pos1)
            self.joint2.publish(self.pos2)
            self.joint3.publish(self.pos3)
            self.joint4.publish(self.pos4)
            self.joint5.publish(self.pos5)

            rospy.wait_for_message('navi2arm', String)

            # gesture 2
            self.pos1 = 0.0
            self.pos2 = 2.09
            self.pos3 = 2.09
            self.pos4 = -0.57
            self.pos5 = -0.4
            self.joint4.publish(self.pos4)
            self.joint1.publish(self.pos1)
            self.joint2.publish(self.pos2)
            self.joint3.publish(self.pos3)
            self.joint5.publish(self.pos5)
            time.sleep(15)
            self.pos3 = 0.0
            self.joint3.publish(self.pos3)
            time.sleep(10)
            self.pos5 = 0.4
            self.joint5.publish(self.pos5)
            time.sleep(5)
            self.joint4.publish(-0.9)
            pub.publish('arm2navi')
            rospy.wait_for_message('navi_finish', String)
            print('countinue')

            # gesture 3
            self.pos1 = 0.0
            self.pos2 = 0.52
            self.pos3 = 0.0
            self.pos4 = 0.52
            self.pos5 = 0.2
            self.joint1.publish(self.pos1)
            self.joint2.publish(self.pos2)
            self.joint4.publish(self.pos4)
            self.joint3.publish(self.pos3)
            time.sleep(10)
            self.joint5.publish(self.pos5)
            rospy.sleep(3)
            break



        def cleanup(self):
            rospy.loginfo("Shutting down robot arm....")

    if __name__=="__main__":
        rospy.init_node('arm')
        try:
            Loop()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

###IV  Summary and Reflection  
Reviewing the process of the whole project, the basic functions can be realized, and the task of grabbing specific object can be accomplished.   
But there are also two more important issues that can be improved.   
The first is that the adjustment of navigation errors can be further improved. Because the depth information of Kinect is not very accurate, I only used a relatively accurate part of the data in the project. I have filtered out the information that can be judged to be not bottle points,.But there are other point clouds that have not been ruled out. For example, information about things next to the bottle is also computed. So the data obtained from the distance is not very accurate, which directly leads to the reduction of the success rate of grabbing objects.  
The second is the posture control of the arm. It would be good to give the arm a specific angle directly.But it would be more accurate if the depth information of Kinect and Tf could be added.  
In a word, I have gained a lot in the course of Robot Software Engineering. I have also found my own interests. Thank you very much for your careful guidance!


