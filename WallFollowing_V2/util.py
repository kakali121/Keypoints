'''
Author       : Karen Li
Date         : 2023-08-11 17:54:58
LastEditors  : Karen Li
LastEditTime : 2023-08-11 18:02:11
FilePath     : /WallFollowing_V2/util.py
Description  : Utility functions
'''

### Import Packages ###
import numpy as np
import cv2

def capture_frame_from_demo(interval: int, demo_video: str, interval_length: int) -> any:
    '''
    description: Read a frame from the demo video at the specified interval
    param       {int} interval: The interval of the frame to be read
    param       {str} demo_video: The path to the demo video
    param       {int} interval_length: The number of frames in a timeline interval
    return      {any} frame: The frame read from the demo video
    '''
    # Create a VideoCapture object to read the video file
    video_cap = cv2.VideoCapture(demo_video)
    # Set a desiered frame position
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, interval*interval_length)
    # Read the frame at the specified position
    ret, frame = video_cap.read()
    # Check if the frame was successfully read
    if not ret:
        print("No frame at interval: ", interval)
        return None
    return frame