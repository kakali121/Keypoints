'''
Author       : Karen Li
Date         : 2023-08-11 17:45:14
LastEditors  : Karen Li
LastEditTime : 2023-08-11 17:53:38
FilePath     : /WallFollowing_V2/imitate.py
Description  : Let robot immitate the behavior of the demo
'''

### Import Packages ###
import numpy as np
import socket
import math
import util_old
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