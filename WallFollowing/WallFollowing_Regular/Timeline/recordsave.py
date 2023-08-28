'''
Author       : Karen Li
Date         : 2023-08-14 14:18:12
LastEditors  : Karen Li
LastEditTime : 2023-08-14 14:48:42
FilePath     : /WallFollowing_V2/Timeline/save2.py
Description  : 
'''

### Import Packages ###
import cv2
import numpy as np

MAX_FRAMES = 2400                      # large numbers will cover the whole video
INTERVAL = 10                          # 13 frames per inverval 
MAX_MATCH_DISTANCE = 40                # match threshold

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create(nfeatures=1000)
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

VIDEO = "trimmed.mp4"                 # The path to the demo video
NAME = "output2.mp4"                  # The path to the output video

if __name__ == "__main__":
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture(VIDEO)
    frames = []

    # Loop through the video frames
    for _ in range(MAX_FRAMES):
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    out = cv2.VideoWriter(NAME, fourcc, fps, (400, 300))

    for frame in frames:
        out.write(frame)
        cv2.imshow("frame", frame)
        # Wait for Esc key to stop
        if cv2.waitKey(1) == ord('q'):
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            break

    for frame in reversed(frames):
        out.write(frame)
        cv2.imshow("frame", frame)
        # Wait for Esc key to stop
        if cv2.waitKey(1) == ord('q'):
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            break

    for frame in frames:
        out.write(frame)
        cv2.imshow("frame", frame)
        # Wait for Esc key to stop
        if cv2.waitKey(1) == ord('q'):
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            break

    for frame in reversed(frames):
        out.write(frame)
        cv2.imshow("frame", frame)
        # Wait for Esc key to stop
        if cv2.waitKey(1) == ord('q'):
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            break

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()
    cap.release()
    out.release()