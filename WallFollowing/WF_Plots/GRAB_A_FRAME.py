"""
Author       : Hanqing Qi
Date         : 2023-09-14 18:59:57
LastEditors  : Hanqing Qi
LastEditTime : 2023-09-15 22:48:29
FilePath     : /WF_Plots/GRAB_A_FRAME.py
Description  : GRAB A FRAME
"""

import cv2

def grab_nth_frame(video_path, n):
    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file is opened successfully.
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return None

    # Set the frame position.
    cap.set(cv2.CAP_PROP_POS_FRAMES, n-1)
    
    # Read the frame.
    ret, frame = cap.read()

    # Check if the frame was read successfully.
    if not ret:
        print("Error: Couldn't read the frame.")
        return None

    # Close the video file.
    cap.release()

    return frame

# Example usage:
video_path = "robot.mp4"
for n in range(350,400):
    frame = grab_nth_frame(video_path, n)
    if frame is not None:
        # cv2.imshow(f"Frame {n}", frame)
        # Save the frame as an pdf file.
        cv2.imwrite(f"frame_{n}.png", frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()