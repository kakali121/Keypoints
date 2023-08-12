import cv2
import numpy as np
import socket
import math

IP_ADDRESS = '192.168.0.204'                           # IP address of the robot
STREAM_URL = "http://192.168.0.204:1234/stream.mjpg"   # video stream url

DES_FILE = "side_demo_kpt_des"                         # demo keypoints and descriptors file
DEMO_VIDEO = "sidedemo.mp4"                            # demo video file
INTERVAL_LENGTH = 12                                   # number of frames in an interval
SKIP_INTERVAL = 5                                      # interval between donkey and carrot
MAX_MATCH_DISTANCE = 30                                # match threshold

ACCUMULATED_Y_RATIO = 0.0                              # accumulated y ratio
V_GAIN = -3                                            # velocity gain
W_GAIN = 300                                           # angular velocity gain

# Create an ORB object and detect keypoints and descriptors
orb = cv2.ORB_create(nfeatures=1000)
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

CONNECT = True

    

if __name__ == "__main__":

    # Create a video capture object to read video stream from camera
    stream_cap = cv2.VideoCapture(STREAM_URL)

    while stream_cap.isOpened():

    ##### Robot Search #####################################################################################################

        # Read robot frame from the video stream
        ret, robot_frame = stream_cap.read()
        if not ret: break
    
        # Detect keypoints and compute descriptors in the captured frame
        ret_keypoints, ret_descriptors, robot_frame = util_old.find_kpt_des(robot_frame, draw=True)

        # Set the robot's keypoints and descriptors to the captured frame's
        robot_keypoints = util_old.convert_keypoints(ret_keypoints)
        robot_descriptors = ret_descriptors

    ##### Donkey Search ####################################################################################################
    
        if not found_donkey:
            donkey_frame, robot_frame, interval = find_donkey(robot_descriptors, robot_keypoints, robot_frame)
            if donkey_frame is None: continue
            found_donkey = True

    ##### Carrot Persue ####################################################################################################
        
        # Load goal image keypoints (Carrot)
        goal_keypoints, goal_descriptors = util_old.load_descriptors(DES_FILE + "/" + DES_FILE + str((interval)+SKIP_INTERVAL) + ".yml")

        # Find the best matching keypoints between the robot and goal image
        rob_goal_matches = bf.match(robot_descriptors, goal_descriptors)
        rob_goal_matches = [m for m in rob_goal_matches if m.distance < MAX_MATCH_DISTANCE]
        print("# Goal kpt matches: " + str(len(rob_goal_matches))) if printer else None
        if len(rob_goal_matches) < 3:
            print("No matches found") if printer else None
            command = 'CMD_MOTOR#00#00#00#00\n'
            if CONNECT: s.send(command.encode('utf-8'))
            continue

        # Find matching keypoints coordinates for retrived and reference keypoints
        robot_xy, carrot_xy = util_old.keypoint_coordinate(rob_goal_matches, robot_keypoints, goal_keypoints)

        # Draw the carrot frame
        robot_frame, carrot_frame = draw_carrot_frame(len(rob_goal_matches), robot_xy, carrot_xy, robot_frame, interval)
        
    ##### Control Analysis ####################################################################################################
        
        q_mx, q_my, q_rx, q_ry, t_mx, t_my, t_rx, t_ry = util_old.pair_analysis(robot_xy, carrot_xy)
        cx_diff = 0
        rx_ratio = 1
        cx_diff = (t_mx - q_mx)
        rx_ratio = t_rx / q_rx
        ry_ratio = t_ry / q_ry
        print("x_diff: " + str(cx_diff) + " rx_ratio: " + str(rx_ratio) + " ry_ratio: " + str(ry_ratio)) if printer else None

    ##### Robot Control ######################################################################################################

        # If the robot is close enough to the carrot (less the 10 pixels), stop
        if abs(cx_diff) < 10: command = 'CMD_MOTOR#00#00#00#00\n'
        else:
            v = V_GAIN * cx_diff + 500
            # Calculate the moving average of the y ratio
            processed_y_ratio = calculate_moving_average_y(ry_ratio)
            print("Processed y ratio: " + str(processed_y_ratio)) if printer else None
            w = (1 - processed_y_ratio) * W_GAIN
            u = np.array([v - w, v + w])
            print("V: " + str(v)) if printer else None
            print("W: " + str(w)) if printer else None
            print("U: " + str(u)) if printer else None
            for i in range(len(u)):
                if u[i] >= 0:
                    u[i] = min(u[i], 800)
                    u[i] = max(u[i], 500)
                else:
                    u[i] = max(u[i], -800)
                    u[i] = min(u[i], -500)
            command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(u[0], u[0], u[1], u[1])
        if CONNECT: s.send(command.encode('utf-8'))
        print(command) if printer else None

        cv2.imshow('Carrot', carrot_frame)   # Display the demo frame
        cv2.imshow('Robot', robot_frame)     # Display the robot frame
        cv2.imshow('Donkey', donkey_frame)   # Display the demo frame
        
        # Wait for Esc key to stop
        if cv2.waitKey(1) == 27:
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            stream_cap.release()
            if CONNECT:
                command = 'CMD_MOTOR#00#00#00#00\n'
                s.send(command.encode('utf-8'))
                s.close()
            break