import util
import cv2
import os
import math
import time
import numpy as np
import socket

IP_ADDRESS = '192.168.0.204'                           # IP address of the robot
STREAM_URL = "http://192.168.0.204:1234/stream.mjpg"   # video stream url

DES_FILE = "side_demo_kpt_des"                         # demo keypoints and descriptors file
DEMO_VIDEO = "demoS.mp4"                               # demo video file

SKIP_INTERVAL = 5                                      # interval to skip to next position
MAX_MATCH_DISTANCE = 50                                # match threshold

# Create an ORB object and detect keypoints and descriptors
orb = cv2.ORB_create(nfeatures=1000)
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


if __name__ == "__main__":
    # Connect to the robot
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP_ADDRESS, 5000))
    print('Connected')
    # Create a video capture object to read video stream from camera
    cap = cv2.VideoCapture(STREAM_URL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Donkey Search ############################################################
        # Detect keypoints and compute descriptors in the captured frame
        ret_keypoints, ret_descriptors, frame = util.find_kpt_des(frame, draw=True)
        # Set the robot's keypoints and descriptors to the captured frame's
        robot_keypoints = util.convert_keypoints(ret_keypoints)
        robot_descriptors = ret_descriptors

        # Find the best matching interval reference
        interval = util.find_best_interval(robot_descriptors, DES_FILE)


