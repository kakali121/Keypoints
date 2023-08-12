'''
Author       : Karen Li
Date         : 2023-08-11 17:45:14
LastEditors  : Hanqing Qi
LastEditTime : 2023-08-12 11:13:50
FilePath     : /WallFollowing_V2/Imitate.py
Description  : Let robot immitate the behavior of the demo
'''

### Import Packages ###
import numpy as np
import socket
import robot
import math
import util
import cv2

### Global Variables ###
IP_ADDRESS = '192.168.0.204'                               # IP address of the robot
STREAMING_URL = "http://192.168.0.204:1234/stream.mjpg"    # Video streaming url

DESCRIPTOR_FILE_PATH = "side_demo_kpt_des"                 # Path to the descriptor files
DEMO_VIDEO = "sidedemo.mp4"                                # Demo video file
INTERVAL_LENGTH = 12                                       # Number of frames in a timeline interval
SKIP_INTERVAL = 5                                          # Interval between donkey and carrot
MAX_MATCH_DISTANCE = 30                                    # The maximum distance between two matched keypoints

ACCUMULATED_Y_RATIO = 0.0                                  # Accumulated y ratio
V_GAIN = -3                                                # Gain of velocity
W_GAIN = 300                                               # Gain of angular velocity

CONNECT_TO_ROBOT = True                                    # Whether to connect to the robot

### Initialization ###

# Create an ORB object and detect keypoints and descriptors
ORB_Object = cv2.ORB_create(nfeatures=1000)
# Create a brute-force matcher object
brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Create a robot object
myrobot = robot.Robot(IP_ADDRESS, CONNECT_TO_ROBOT)
# Create a VideoCapture object to read the video stream
streaming_video = cv2.VideoCapture(STREAMING_URL) 

### Main Loop ###
while streaming_video.isOpened():
    # Read a frame from the video stream
    ret, frame = streaming_video.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Detect keypoints and descriptors of the frame
    kpt_frame, des_frame = ORB_Object.detectAndCompute(frame, None)
    # Read the timeline interval from the demo video
    interval = util.read_interval(DESCRIPTOR_FILE_PATH)
    # Read the frame from the demo video at the specified interval
    frame_demo = util.capture_frame_from_demo(interval, DEMO_VIDEO, INTERVAL_LENGTH)
    # Detect keypoints and descriptors of the frame from the demo video
    kpt_demo, des_demo = ORB_Object.detectAndCompute(frame_demo, None)
    # Match the keypoints from the frame and the demo
    matches = brute_force_matcher.match(des_frame, des_demo)
    # Sort the matches by distance
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Filter the matches by distance
    matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]
    # Calculate the ratio of the y coordinate of the matched keypoints
    y_ratio = [kpt_frame[m.queryIdx].pt[1] / kpt_demo[m.trainIdx].pt[1] for m in matches]
    # Calculate the average of the y ratio
    y_ratio_avg = sum(y_ratio) / len(y_ratio)
    # Calculate the angular velocity
    ω = W_GAIN * (y_ratio_avg - ACCUMULATED_Y_RATIO)
    # Calculate the linear velocity
    v = V_GAIN * (1 - math.exp(-abs(ω)))
    # Move the robot
    myrobot.move(v, ω)
    # Update the accumulated y ratio
    ACCUMULATED_Y_RATIO = y_ratio_avg
    # Display the frame
    cv2.imshow('frame', frame)
    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break