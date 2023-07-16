#!/usr/bin/env python3

import rospy
import threading, requests, time
import math
import cv2
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped , Vector3
from mavros_msgs.msg import State , AttitudeTarget
from mavros_msgs.srv import SetMode, CommandBool , SetModeRequest , SetMavFrame , SetMavFrameRequest
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import NavSatFix , Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
#aruco
from utils import ARUCO_DICT, aruco_display




class PID:
    def __init__(self, kp, ki, kd, max_output, min_output, max_integ, min_integ, sample_time):
        self.kp = kp  # Proportional Gain
        self.ki = ki  # Integral Gain
        self.kd = kd  # Derivative Gain
        self.max_output = max_output  # Maximum Output
        self.min_output = min_output  # Minimum Output
        self.max_integ = max_integ  # Maximum Integral Term
        self.min_integ = min_integ  # Minimum Integral Term
        self.sample_time = sample_time  # Sample Time

        self.target = 0.0  # Target Value
        self.integ = 0.0  # Integral Term
        self.last_error = 0.0  # Last Error
        self.last_time = rospy.Time.now()  # Last Time

    def update(self, feedback_value):
        error = self.target - feedback_value  # Error
        dt = (rospy.Time.now() - self.last_time).to_sec()  # Time Step

        # Proportional Term
        P = self.kp * error

        # Integral Term
        self.integ += error * dt
        self.integ = max(self.min_integ, min(self.max_integ, self.integ))
        I = self.ki * self.integ

        # Derivative Term
        D = self.kd * (error - self.last_error) / dt

        # PID Output
        output = P + I + D
        output = max(self.min_output, min(self.max_output, output))

        # Update Last Error and Last Time
        self.last_error = error
        self.last_time = rospy.Time.now()

        return output

class MoveDrone:
    def __init__(self):
        rospy.init_node('move_drone', anonymous=True)

        # Initialize ROS Subscriber
        rospy.Subscriber('/mavros/state', State, self.state_cb)
        #rospy.Subscriber('/mavros/local_position/odom', Odometry, self.position_cb)
        rospy.Subscriber("/mavros/global_position/global" , NavSatFix , self.callBaclkGlobalPosition)
        rospy.Subscriber("/iris/camera/rgb/image_raw", Image, self.image_callback)
        rospy.Subscriber('/laser/scan', LaserScan, self.laserScan_callback)


        # Initialize ROS Publisher
        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
        self.alt_pub = rospy.Publisher('/mavros/setpoint_position/rel_alt', Float32, queue_size=10)
        self.att_pub = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)



        
        # Initialize ROS Service
        self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.flight_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)

          # frame변경
        self.set_mav_frame =rospy.ServiceProxy("mavros/setpoint_velocity/mav_frame", SetMavFrame)     


        # Initialize Variables
        self.current_state = State()
        self.current_pose = None
        self.current_global_position = None
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = "home"
        self.target_pose.pose.position.x = 10.0
        self.target_pose.pose.position.y = 10.0
        self.target_pose.pose.position.z = 0.0
        self.offb_set_mode = SetModeRequest()
        self.offb_set_mode.custom_mode = 'OFFBOARD'
        
        self.frame_set_mode = SetMavFrameRequest()
        self.frame_set_mode.mav_frame = 8

        self.att_msg = AttitudeTarget()
        self.att_msg.header.stamp = rospy.Time.now()
        self.att_msg.type_mask = 128  # 롤, 피치, 요륙 속도 제어

        self.arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_ARUCO_ORIGINAL"])
        self.arucoParams = cv2.aruco.DetectorParameters_create()



        # tracking 관련 target_lat = 35.8934302
        self.target_lat = None    
        self.target_lng = None     

        # CvBridge
        self.bridge = CvBridge()  

        # Initialize PID Controllers
        self.x_pid = PID(0.2, 0.01, 0.01, 0.3, -0.3, 1.0, -1.0, 0.1)
        self.y_pid = PID(0.2, 0.01, 0.01, 0.1, -0.1, 1.0, -1.0, 0.1)
        self.z_pid = PID(0.2, 0.01, 0.01, 0.1, -0.1, 1.0, -1.0, 0.1)

        self.target_center_x = None
        self.target_center_ｙ = None

        self.image_center_x = None
        self.image_center_y = None
        
        # Initialize laserScanArray
        self.laserScanVal = None
        

        self.x_output = 0.0

        self.alignCheck = 0.0


    def laserScan_callback(self,msg):
        self.laserScanVal = msg
        #print(msg.ranges[179]) # front

    def show_image(self,img):
        cv2.imshow("Image Window", img)
        
        cv2.waitKey(1)

    # Define a callback for the Image message
    def image_callback(self,img_msg):
        # log some info about the image topic
        #rospy.loginfo(img_msg.header)

        # Try to convert the ROS Image message to a CV2 Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8") # color
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        h, w, _ = cv_image.shape

        # image center x , y
        self.image_center_y = int(h/2)
        self.image_center_x = int(w/2)

        #width=1000
        #height = int(width*(h/w))
        #cv_image = cv2.resize(cv_image, (width, height), interpolation=cv2.INTER_CUBIC)
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.arucoDict, parameters=self.arucoParams)

        (m_x , m_y , detected_markers) = aruco_display(corners, ids, rejected, cv_image)
        
        self.target_center_x = m_x
        self.target_center_ｙ = m_y

        cv2.circle(detected_markers, (int(w/2), int(h/2)), 2, (255, 0, 0), -1)

        cv2.circle(detected_markers, (m_x, m_y), 2, (0, 0, 255), -1)

        # Show the converted image
        self.show_image(detected_markers)


    def callBaclkGlobalPosition(self,data):
        #print(data)     
        self.current_global_position = data

        #sub_topics_ready['global_pos'] = True


    def setOffboard(self):
        last_req = rospy.Time.now()
        
        while(not rospy.is_shutdown()):
            if(self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if((self.flight_mode_service.call(self.offb_set_mode).mode_sent == True)):
                    #self.flight_mode_service.call(self.offb_set_mode)
                    rospy.loginfo("OFFBOARD enabled")
                    break
            # else:
            #     break

    def state_cb(self, state_msg):
        self.current_state = state_msg

    def position_cb(self, odom_msg):
        self.current_pose = odom_msg.pose.pose

    def arm(self):
        rospy.loginfo("Arming Drone")       
        while not rospy.is_shutdown():
            if self.current_state.armed:
                break
            self.arm_service(True)
            rospy.sleep(1)
      

    def takeoff(self):
        rospy.loginfo("Taking off")
        while not rospy.is_shutdown():
            # print(self.current_state.mode)
            # if self.current_pose.position.z >= 3.0:
            #     break
            # self.alt_pub.publish(3.0)

            vel_msg = Twist()
            vel_msg.linear.x = 1500.0
            vel_msg.linear.y = 0.0
            vel_msg.linear.z = 2100.0
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.z = 0.0
            self.vel_pub.publish(vel_msg)
            rospy.sleep(0.1)

    def move_drone(self):
        rospy.loginfo("Moving Drone")

        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            
            if (self.set_mav_frame.call(self.frame_set_mode).success == False):
                rospy.loginfo("Waiting for mav frame update...")
                continue

            # if self.current_pose is None:
            #     rospy.loginfo("Waiting for pose update...")
            #     continue

            # if self.current_global_position is None:
            #     rospy.loginfo("Waiting for global position update...")
            #     continue

            # if self.target_lat is None:
            #     rospy.loginfo("Waiting for target position update...")
            #     continue
            # Update Target Position
            # x_error = self.target_pose.pose.position.x - self.current_pose.position.x
            # y_error = self.target_pose.pose.position.y - self.current_pose.position.y
            # z_error = self.target_pose.pose.position.z - self.current_pose.position.z
            
            #target_lat = 35.8934302
            #target_lng = 128.6134666

            # x_error = target_lat -self.current_global_position.latitude
            # y_error = target_lng -self.current_global_position.longitude
            # x_error , y_error = self.distance_on_xy(self.target_lat , self.target_lng ,self.current_global_position.latitude, self.current_global_position.longitude)   
                        
            if self.target_center_y is None:
                rospy.loginfo("Waiting for target position update...")
                continue

            if self.image_center_x is None:
                rospy.loginfo("Waiting for target position update...")
                continue

            print(self.target_center_x)
            y_error = self.target_center_x -self.image_center_x
            z_error = self.target_center_y - self.image_center_y

            #print("target : ",self.target_center_x , self.target_center_y)

            #print("center : ",self.image_center_x , self.image_center_y)


#            print(x_error , y_error)
            
            if(abs(y_error) <10):
                y_error = 0
            
            if(abs(z_error) < 10):
                z_error = 0

            
            y_output = self.y_pid.update(y_error)
            z_output = self.z_pid.update(z_error)

            
            if(self.target_center_x == -1):
                y_output = 0.0
                
            if(self.target_center_y == -1):
                z_output = 0.0

            if(abs(y_error) < 10 and abs(z_error) < 10 ):
                self.alignCheck = self.alignCheck +1


            # align succeed
            if(self.alignCheck >60 and self.laserScanVal != None):
                #y_output = z_output = 0.0
                if(self.laserScanVal.ranges[179] > 1.2):
                    self.x_output = 0.1
                else:
                    print("drop pizza box")
                    self.x_output = 0.0


            #if(self.laserScanVal != None):
            #    print(self.laserScanVal.ranges[179])
#             print(x_output , y_output)
            # Publish Velocity
            vel_msg = Twist()
            #vel_msg.header.stamp = rospy.Time.now()
            vel_msg.linear.x = self.x_output
            vel_msg.linear.y = y_output
            vel_msg.linear.z = z_output
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.z = 0.0
            self.vel_pub.publish(vel_msg)

            # # 롤, 피치, 요륙 속도 값을 설정합니다.
            # self.att_msg.body_rate.x = 0.0  # roll
            # self.att_msg.body_rate.y = -0.2 # pitch
            # self.att_msg.body_rate.z = 0.0  # yaw

            # # 스로틀 값을 설정합니다.
            # self.att_msg.thrust = 0.4 #

            # # AttitudeTarget 메시지를 발행합니다.
            # self.att_pub.publish(self.att_msg)

            rate.sleep()

    # tracking 관련 함수
    def distance_on_xy(self, lat1, lon1, lat2, lon2):
        R = 6371000  # 지구의 반지름 (m)
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        dx = R * math.cos(lat1_rad) * (lon2_rad - lon1_rad)
        dy = R * (lat2_rad - lat1_rad)
        return dx, dy 

if __name__ == "__main__":

    move_drone = MoveDrone()

     # Arm Drone
    #move_drone.arm()

    # Takeoff Drone
    #move_drone.takeoff()
    #rospy.sleep(2.0)
    #move_drone.setOffboard()
    move_drone.move_drone()