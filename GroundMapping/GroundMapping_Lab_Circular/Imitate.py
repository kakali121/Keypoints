'''
Author       : Karen Li
Date         : 2023-08-11 17:45:14
LastEditors  : Karen Li
LastEditTime : 2023-09-01 16:00:28
FilePath     : /WallFollowing_Corner/Imitate.py
Description  : Let robot immitate the behavior of the demo
'''

### Import Packages ###
from WallTraker import WallTraker
import matplotlib.pyplot as plt
import Robot
import math
import cv2

### Global Variables ###
IP_ADDRESS = '192.168.0.204'     # IP address of the robot
STREAMING_URL = "http://192.168.0.204:1234/stream.mjpg"  # Video streaming url

TOTAL_INTERVALS = 200            # Total number of intervals in the demo video
INTERVAL_LENGTH = 12             # Number of frames in a timeline interval
SKIP_INTERVAL = 3                # Interval between donkey and carrot

V_GAIN = 3                       # Gain of velocity
W_GAIN = 400                     # Gain of angular velocity

CONNECT_TO_ROBOT = True          # Whether to connect to the robot
V_VALUES = []                    # A list of linear velocities
ω_VALUES = []                    # A list of angular velocities
NUM_MATCH = []                   # A list of number of matches

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
# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20
out1 = cv2.VideoWriter("robot.mp4", fourcc, fps, (400, 300))
out2 = cv2.VideoWriter("carrot.mp4", fourcc, fps, (400, 300))

def plot_speeds():
    # # Plot v values
    # plt.figure()
    # plt.plot(V_VALUES)
    # plt.title('Velocity (v) over Time')
    # plt.xlabel('Time')
    # plt.ylabel('Velocity (v)')
    # plt.grid(True)
    # plt.savefig("v_plot.png")
    # plt.show()

    # Plot ω values
    plt.figure(1)  # Create a new figure window
    plt.plot(ω_VALUES)
    plt.title('Angular Velocity (ω) over Time')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity (ω)')
    plt.grid(True)
    plt.savefig("omega_plot.png")

    # Plot number of matches
    plt.figure(2)  # Create another new figure window
    plt.plot(NUM_MATCH)
    plt.title('Number of Matches over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Matches')
    plt.grid(True)
    plt.savefig("match_plot.png")

    # Now show both figures
    plt.show()

### Main Loop ###
try:
    while streaming_video.isOpened():
        if not position == -1: print("Going to interval: ", position)
        x_diff, processed_y_ratio, num_match, lost = wall_tracker.chase_carrot()
        robot_frame, carrot_frame = wall_tracker.show_all_frames()
        out1.write(robot_frame)
        out2.write(carrot_frame)
        v = V_GAIN * x_diff  # Compute the linear velocity
        ω = W_GAIN * (1 - processed_y_ratio)  # Compute the angular velocity
        print("x_diff: ", x_diff)
        print("processed_y_ratio: ", processed_y_ratio)
        print("v: ", v, "\nω: ", ω)
        V_VALUES.append(v)
        ω_VALUES.append(ω)
        NUM_MATCH.append(num_match)


        if abs(x_diff) < 10 and not lost: # If the robot is close enough to the carrot
            position = wall_tracker.next_carrot() # Go to the next carrot
            lost_count = 0 # Reset the lost count
        if math.isnan(v) or math.isnan(ω):  # If the velocity is NaN, stop the robot
            raise Exception("Velocity is NaN. Exiting ...")
        if lost: # If the robot lost the wall
            if lost_count > 20: raise Exception("Lost too many times. Exiting ...")
            v, ω = 0, 0 # Stop the robot
            lost_count += 1
            print("Lost count: ", lost_count)
        
        myrobot.move_legacy(v, ω)

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
    plot_speeds()


except (Exception, KeyboardInterrupt) as e:
    # De-allocate any associated memory usage
    print("Get exception: ", e)
    cv2.destroyAllWindows()
    streaming_video.release()
    myrobot.disconnect()
    plot_speeds()
