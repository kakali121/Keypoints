import util
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


def show_reference_demo(interval, video_file=DEMO_VIDEO, length=INTERVAL_LENGTH):
    # Create a VideoCapture object to read the video file
    video_cap = cv2.VideoCapture(video_file)
    # Set a desiered frame position
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, interval*length)
    # Read the frame at the specified position
    ret, frame = video_cap.read()
    # Check if the frame was successfully read
    if not ret:
        print("No frame at interval ", interval)
        return None
    return frame


def draw_carrot_frame(k, robot_xy, carrot_xy, robot_frame, interval, skip=SKIP_INTERVAL):
    # Find the goal image
    carrot_frame = show_reference_demo((interval+skip))
    if carrot_frame is None: return None, None
    # Draw the matching keypoints
    for i in range(k):
        carrot_center = (int(carrot_xy[i][0]), int(carrot_xy[i][1]))
        robot_center = (int(robot_xy[i][0]), int(robot_xy[i][1]))    
        carrot_frame = cv2.circle(carrot_frame, carrot_center, 3, (255, 0, 0), -1)                             # blue dot
        carrot_frame = cv2.circle(carrot_frame, robot_center, 3, (0, 0, 255), -1)                              # red dot
        carrot_frame = cv2.arrowedLine(carrot_frame, carrot_center, robot_center, (0, 255, 255), thickness=1)  # yellow line
        robot_frame = cv2.circle(robot_frame, robot_center, 3, (0, 0, 255), -1)                                # red dot
        robot_frame = cv2.circle(robot_frame, carrot_center, 3, (255, 0, 0), -1)                               # blue dot
        robot_frame = cv2.arrowedLine(robot_frame, robot_center, carrot_center, (0, 255, 255), thickness=1)    # yellow line
    return robot_frame, carrot_frame


def draw_donkey_frame(k, robot_xy, donkey_xy, robot_frame, interval):
    # Find the best matching reference image
    donkey_frame = show_reference_demo(interval)
    if donkey_frame is None: return None, None, None
    # Draw the matching keypoints
    for i in range(k):
        donkey_center = (int(donkey_xy[i][0]), int(donkey_xy[i][1]))
        robot_center1 = (int(robot_xy[i][0]), int(robot_xy[i][1]))
        donkey_frame = cv2.circle(donkey_frame, donkey_center, 3, (255, 0, 255), -1)                 # pink dot
        robot_frame = cv2.circle(robot_frame, robot_center1, 3, (255, 0, 255), -1)   
    return donkey_frame, robot_frame


def find_donkey(robot_descriptors, robot_keypoints, robot_frame, file=DES_FILE):
    # Find the best matching interval reference
    possible_intervals = []
    for i in range(50):
        possible_interval = util_old.find_best_interval(robot_descriptors, file)
        if possible_interval is None: break
        possible_intervals.append(possible_interval)
    interval = max(set(possible_intervals), key = possible_intervals.count) # Most common interval
    print("Best matching interval: " + str(interval))
    # Load reference image keypoints (Donkey)
    ref_keypoints, ref_descriptors = util_old.load_descriptors(file + "/" + file + str(interval) + ".yml")
    # Find the best matching keypoints between the robot and reference image
    rob_ref_matches = bf.match(robot_descriptors, ref_descriptors)
    print("# Ref kpt matches: " + str(len(rob_ref_matches)))
    # Find matching keypoints coordinates for retrived and reference keypoints
    robot_xy, donkey_xy = util_old.keypoint_coordinate(rob_ref_matches, robot_keypoints, ref_keypoints)
    # Draw the donkey frame
    donkey_frame, robot_frame = draw_donkey_frame(len(rob_ref_matches), robot_xy, donkey_xy, robot_frame, interval)
    return donkey_frame, robot_frame, interval


def calculate_moving_average_y(new_y_ratio: float) -> float:
    '''
    description: Function to calculate the moving average of the y ratio
    param       {float} new_y_ration: The current y ratio
    return      {float} The moving average of the y ratio
    '''
    global ACCUMULATED_Y_RATIO
    if ACCUMULATED_Y_RATIO == 0: # If the accumulated y ratio is 0, set it to the current y ratio
        ACCUMULATED_Y_RATIO = new_y_ratio
    else: 
        # Calculate the difference between the current y ratio and the accumulated y ratio
        y_ratio_diff = abs((new_y_ratio - ACCUMULATED_Y_RATIO) / ACCUMULATED_Y_RATIO) 
        if y_ratio_diff > 10: # If the difference is too big, discard the current y ratio
            print("Warning: Broken Match!")
            print("Discard y ratio: " + str(new_y_ratio))
            return ACCUMULATED_Y_RATIO
        # The dynamic gain is the exponential of the difference
        dynamic_gain = 1/math.exp(y_ratio_diff) 
        # Calculate the new accumulated y ratio
        ACCUMULATED_Y_RATIO = ACCUMULATED_Y_RATIO * (1-dynamic_gain) + new_y_ratio * dynamic_gain
    return ACCUMULATED_Y_RATIO

    

if __name__ == "__main__":

    if CONNECT:
        # Connect to the robot
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((IP_ADDRESS, 5000))
        print('Connected')

    found_donkey = False

    # Create a video capture object to read video stream from camera
    stream_cap = cv2.VideoCapture(STREAM_URL)
    counter = 0
    printer = False

    while stream_cap.isOpened():
        counter += 1    
        if counter % 5 == 0: printer = True
        else: printer = False

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