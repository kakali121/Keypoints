'''
Author       : Karen Li
Date         : 2023-08-11 17:45:14
LastEditors  : Karen Li
LastEditTime : 2023-08-14 13:16:19
FilePath     : /WallFollowing_V2/imitate.py
Description  : Let robot immitate the behavior of the demo
'''

### Import Packages ###
from WallTraker import WallTraker
import Robot
import math
import cv2

### Global Variables ###
IP_ADDRESS = '192.168.0.204'     # IP address of the robot
STREAMING_URL = "http://192.168.0.204:1234/stream.mjpg"  # Video streaming url

TOTAL_INTERVALS = 270            # Total number of intervals in the demo video
INTERVAL_LENGTH = 10             # Number of frames in a timeline interval
SKIP_INTERVAL = 6                # Interval between donkey and carrot

V_GAIN = -3                      # Gain of velocity
W_GAIN = 500                     # Gain of angular velocity

CONNECT_TO_ROBOT = True          # Whether to connect to the robot

### Initialization ###
# Create a robot object
myrobot = Robot.Robot(IP_ADDRESS, CONNECT_TO_ROBOT)
# Create a VideoCapture object to read the video stream
streaming_video = cv2.VideoCapture(STREAMING_URL)
# Create a wall tracker object
ret, robot_frame = streaming_video.read()  # Take a frame from the video stream
wall_tracker = WallTraker(robot_frame, TOTAL_INTERVALS, INTERVAL_LENGTH, SKIP_INTERVAL)
# Initialize the counter
position = -1      # The current interval
lost_count = 0     # The number of times the robot lost the wall

### Main Loop ###
try:
    while streaming_video.isOpened():
        if not position == -1: print("Going to interval: ", position)
        x_diff, processed_y_ratio, lost = wall_tracker.chase_carrot()
        wall_tracker.show_all_frames()
        v = V_GAIN * x_diff  # Compute the linear velocity
        ω = W_GAIN * (1 - processed_y_ratio)  # Compute the angular velocity
        print("x_diff: ", x_diff)
        print("processed_y_ratio: ", processed_y_ratio)
        print("v: ", v, "\nω: ", ω)

        if abs(x_diff) < 5 and not lost: # If the robot is close enough to the carrot
            position = wall_tracker.next_carrot() # Go to the next carrot
            lost_count = 0 # Reset the lost count
        if math.isnan(v) or math.isnan(ω):  # If the velocity is NaN, stop the robot
            raise Exception("Velocity is NaN. Exiting ...")
        if lost: # If the robot lost the wall
            if lost_count > 20: raise Exception("Lost too many times. Exiting ...")
            v, ω = 0, 0 # Stop the robot
            lost_count += 1
            print("Lost count: ", lost_count)
        
        myrobot.move(v, ω)

        ret, robot_frame = streaming_video.read()  # Take a frame from the video stream
        if not ret: raise Exception("Can't receive frame (stream end?). Exiting ...")
        wall_tracker.update_robot(robot_frame)

        if cv2.waitKey(1) == 27 or position == TOTAL_INTERVALS:
            if position == TOTAL_INTERVALS: print("Finish!")
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            streaming_video.release()
            myrobot.disconnect()
            break

except (Exception, KeyboardInterrupt) as e:
    # De-allocate any associated memory usage
    print("Get exception: ", e)
    cv2.destroyAllWindows()
    streaming_video.release()
    myrobot.disconnect()
