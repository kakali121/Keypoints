"""
Author       : Karen Li
Date         : 2023-08-11 17:45:14
LastEditors  : Hanqing Qi
LastEditTime : 2023-09-07 18:08:01
FilePath     : /WallFollowing_Lab_Corner_2/Imitate.py
Description  : Let robot immitate the behavior of the demo
"""

### Import Packages ###
from WallTraker import WallTraker
import matplotlib.pyplot as plt
import Robot
import math
import cv2

### Global Variables ###
IP_ADDRESS = "192.168.0.204"  # IP address of the robot
STREAMING_URL = "http://192.168.0.204:1234/stream.mjpg"  # Video streaming url

# TOTAL_INTERVALS = 390            # Total number of intervals in the demo video
# INTERVAL_LENGTH = 12             # Number of frames in a timeline interval
# SKIP_INTERVAL = 2                # Interval between donkey and carrot

DEMO_VIDEO = "corner.mp4"  # name of the demo video
TOTAL_INTERVALS = 600  # Total number of intervals in the demo video
INTERVAL_LENGTH = 10  # Number of frames in a timeline interval
SKIP_INTERVAL = 4  # Interval between donkey and carrot

MAX_HUMMING_DISTANCE = 50  # Max humming distance
MIN_NUM_MATCHES = 10  # Min number of matches
λ = 10  # λ of the dynamic gain

V_GAIN = 3  # Gain of velocity
W_GAIN = 400  # Gain of angular velocity

CONNECT_TO_ROBOT = True  # Whether to connect to the robot

### Lists for Plotting ###
V_VALUES = []  # A list of linear velocities
ω_VALUES = []  # A list of angular velocities
NUM_MATCH = []  # A list of number of matches
RAW_ω = []  # A list of raw ellipse ratios

### Debug Constants ###
debug_dynamic_gain = []  # The dynamic gain of y ratio

### Initialization ###
# Create a robot object
myrobot = Robot.Robot(IP_ADDRESS, CONNECT_TO_ROBOT)
# Create a VideoCapture object to read the video stream
streaming_video = cv2.VideoCapture(STREAMING_URL)
# Create a wall tracker object
ret, robot_frame = streaming_video.read()  # Take a frame from the video stream
wall_tracker = WallTraker(robot_frame, TOTAL_INTERVALS, INTERVAL_LENGTH, SKIP_INTERVAL, MAX_HUMMING_DISTANCE, DEMO_VIDEO)
# Initialize the counter
position = -1  # The current interval
lost_count = 0  # The number of times the robot lost the wall
# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 20
out1 = cv2.VideoWriter("./Results/robot.mp4", fourcc, fps, (400, 300))
out2 = cv2.VideoWriter("./Results/carrot.mp4", fourcc, fps, (400, 300))


def plot_speeds():
    # # Plot v values
    # plt.figure(0)
    # plt.plot(V_VALUES)
    # plt.title('Velocity (v) over Time')
    # plt.xlabel('Time')
    # plt.ylabel('Velocity (v)')
    # plt.grid(True)
    # plt.savefig("v_plot.pdf")

    # Plot ω values with raw ellipse ratio on the same plot
    plt.figure(1)  # Create a new figure window
    plt.plot(ω_VALUES)
    plt.title("Angular Velocity (ω) over Time")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.grid(True)
    plt.savefig("./Results/omega_plot.pdf")

    # Plot number of matches
    plt.figure(2)  # Create another new figure window
    plt.plot(NUM_MATCH)
    plt.title("Number of Matches over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Matches")
    plt.grid(True)
    plt.savefig("./Results/match_plot.pdf")

    # Plot raw ellipse ratio
    plt.figure(3)  # Create another new figure window
    plt.plot(RAW_ω)
    # Plot the line y=1
    plt.plot(
        [
            0,
            len(RAW_ω),
        ],
        [1, 1],
        color="red",
        linestyle="dashed",
        linewidth=1,
    )
    plt.title("Raw Ellipse Ratio over Time")
    plt.xlabel("Time")
    plt.ylabel("Raw Ellipse Ratio")
    plt.grid(True)
    plt.savefig("./Results/raw_ellipse_plot.pdf")

    # Plot dynamic gain
    plt.figure(4)  # Create another new figure window
    plt.plot(debug_dynamic_gain)
    plt.title("Dynamic Gain over Time")
    plt.xlabel("Time")
    plt.ylabel("Dynamic Gain")
    plt.grid(True)
    plt.savefig("./Results/dynamic_gain_plot.pdf")

    # Now show all figures
    plt.show()


### Main Loop ###
try:
    while streaming_video.isOpened():
        if not position == -1:
            print("Going to interval: ", position)
        (
            x_diff,
            processed_y_ratio,
            num_match,
            lost,
            raw_ellipse_ratio,
            debug_dynamic_gain,
        ) = wall_tracker.chase_carrot()
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
        RAW_ω.append(raw_ellipse_ratio)

        if abs(x_diff) < 10 and not lost:  # If the robot is close enough to the carrot
            position = wall_tracker.next_carrot()  # Go to the next carrot
            lost_count = 0  # Reset the lost count
        if math.isnan(v) or math.isnan(ω):  # If the velocity is NaN, stop the robot
            raise Exception("Velocity is NaN. Exiting ...")
        if lost:  # If the robot lost the wall
            if lost_count > 20:
                raise Exception("Lost too many times. Exiting ...")
            v, ω = 0, 0  # Stop the robot
            lost_count += 1
            print("Lost count: ", lost_count)

        myrobot.move_legacy(v, ω)

        ret, robot_frame = streaming_video.read()  # Take a frame from the video stream
        if not ret:
            raise Exception("Can't receive frame (stream end?). Exiting ...")
        wall_tracker.update_robot(robot_frame)

        if cv2.waitKey(1) == 27 or position == TOTAL_INTERVALS:
            if position == TOTAL_INTERVALS:
                print("Finish!")
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            streaming_video.release()
            myrobot.disconnect()
            break
    # plot_speeds()
    # Close the video writer
    out1.release()
    out2.release()


except (Exception, KeyboardInterrupt) as e:
    # De-allocate any associated memory usage
    print("Get exception: ", e)
    cv2.destroyAllWindows()
    streaming_video.release()
    myrobot.disconnect()
    # plot_speeds()
    # Close the video writer
    out1.release()
    out2.release()
