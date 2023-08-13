'''
Author       : Karen Li
Date         : 2023-08-12 17:43:33
LastEditors  : Karen Li
LastEditTime : 2023-08-12 18:06:16
FilePath     : /WallFollowing_V2/save_a_frame.py
Description  : 
'''
import cv2
import os
import math
import numpy as np

IMAGE_FILE = 'image2.jpg'       # image file
VIDEO_FILE = 'sidedemo.mp4'     # video file
POSITION = 2060                 # frame position

if __name__ == '__main__':
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture(VIDEO_FILE)
    # Set a desiered frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, POSITION)
    # Read the frame at the specified position
    ret, frame = cap.read()
    # Save the frame as an image file
    cv2.imwrite(IMAGE_FILE, frame)
    # Display the frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()