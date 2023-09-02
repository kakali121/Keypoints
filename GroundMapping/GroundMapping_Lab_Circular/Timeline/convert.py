import cv2 
import sys
import time
from NatNetClient import NatNetClient
from util import quaternion_to_euler_angle_vectorized1

positions = {}
rotations = {}

FILENAME = 'raw.mp4'
NAME = 'circular.mp4'

if __name__ == "__main__":
    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    cap = cv2.VideoCapture(FILENAME)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    out = cv2.VideoWriter(NAME, fourcc, fps, (640, 480))
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if not ret: break
        out.write(frame)
        cv2.imshow("frame", frame)
        # Wait for Esc key to stop
        if cv2.waitKey(1) == ord('q'):
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            break
