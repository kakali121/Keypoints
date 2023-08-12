'''
Author       : Karen Li
Date         : 2023-08-11 17:45:14
LastEditors  : Karen Li
LastEditTime : 2023-08-12 19:04:35
FilePath     : /WallFollowing_V2/imitate.py
Description  : Let robot immitate the behavior of the demo
'''

### Import Packages ###
from Wall_Traker import Wall_Traker
import numpy as np
import socket
import Robot
import math
import cv2

### Global Variables ###
IP_ADDRESS = '192.168.0.204'                               # IP address of the robot
STREAMING_URL = "http://192.168.0.204:1234/stream.mjpg"    # Video streaming url

TOTAL_INTERVALS = 200                                      # Total number of intervals in the demo video
INTERVAL_LENGTH = 12                                       # Number of frames in a timeline interval
SKIP_INTERVAL = 5                                          # Interval between donkey and carrot

V_GAIN = -3                                                # Gain of velocity
W_GAIN = 300                                               # Gain of angular velocity

CONNECT_TO_ROBOT = True                                    # Whether to connect to the robot

### Initialization ###

# Create a robot object
myrobot = Robot.Robot(IP_ADDRESS, CONNECT_TO_ROBOT)
# Create a VideoCapture object to read the video stream
streaming_video = cv2.VideoCapture(STREAMING_URL) 
# Create a wall tracker object
ret, robot_frame = streaming_video.read() # Take a frame from the video stream
wall_tracker = Wall_Traker(robot_frame, TOTAL_INTERVALS, INTERVAL_LENGTH, SKIP_INTERVAL)

### Main Loop ###
while streaming_video.isOpened():
    x_diff, processed_y_ratio = wall_tracker.chase_carrot()
    v = V_GAIN * x_diff # Compute the linear velocity
    ω = W_GAIN * (1-processed_y_ratio) # Compute the angular velocity
    if math.isnan(v) or math.isnan(ω): # If the velocity is NaN, stop the robot
        print("v: ", v)
        print("ω: ", ω)
        break
    if abs(x_diff) < 5:
        Wall_Traker.update_carrot()
    myrobot.move(v, ω)
    ret, robot_frame = streaming_video.read() # Take a frame from the video stream
    Wall_Traker.update_robot(robot_frame)

    