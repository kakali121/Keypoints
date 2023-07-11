import cv2
import os
import math
import numpy as np

IMAGE_FILE = 'image.jpg'    # image file
VIDEO_FILE = 'demo.mp4'     # video file
POSITION = 178              # frame position

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
    
