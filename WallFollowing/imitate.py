import util
import cv2
import numpy as np
import socket

IP_ADDRESS = '192.168.0.204'                           # IP address of the robot
STREAM_URL = "http://192.168.0.204:1234/stream.mjpg"   # video stream url

DES_FILE = "side_demo_kpt_des"                         # demo keypoints and descriptors file
DEMO_VIDEO = "demoS.mp4"                               # demo video file
INTERVAL_LENGTH = 13                                         # number of intervals

SKIP_INTERVAL = 3                                      # interval to skip to next position
MAX_MATCH_DISTANCE = 40                                # match threshold

# Create an ORB object and detect keypoints and descriptors
orb = cv2.ORB_create(nfeatures=1000)
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def show_reference_demo(interval, video_file=DEMO_VIDEO, length=INTERVAL_LENGTH):
    # Create a VideoCapture object to read the video file
    video_cap = cv2.VideoCapture(video_file)
    # Set a desiered frame position
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, interval*INTERVAL_LENGTH)
    # Read the frame at the specified position
    retu, capture = video_cap.read()
    # Check if the frame was successfully read
    if not retu:
        print("No frame at interval ", interval)
        return None
    return capture


def find_donkey(robot_descriptors, robot_keypoints, robot_frame, file=DES_FILE):
    possible_intervals = []
    # Find the best matching interval reference
    for i in range(10):
        possible_interval = util.find_best_interval(robot_descriptors, file)
        if possible_interval is None: break
        possible_intervals.append(possible_interval)
    # Find most common interval
    interval = max(set(possible_intervals), key = possible_intervals.count)
    print("Best matching interval: " + str(interval))
    # Load reference image keypoints (Donkey)
    ref_keypoints, ref_descriptors = util.load_descriptors(file + "/" + file + str(interval) + ".yml")
    # Find the best matching keypoints between the robot and reference image
    rob_ref_matches = bf.match(robot_descriptors, ref_descriptors)
    print("# Ref kpt matches: " + str(len(rob_ref_matches)))
    # Find matching keypoints coordinates for retrived and reference keypoints
    robot_xy1, donkey_xy = util.keypoint_coordinate(rob_ref_matches, robot_keypoints, ref_keypoints)
    # Find the best matching reference image
    donkey_frame = show_reference_demo(interval)
    # Draw the matching keypoints
    for i in range(len(rob_ref_matches)):
        donkey_center = (int(donkey_xy[i][0]), int(donkey_xy[i][1]))
        robot_center1 = (int(robot_xy1[i][0]), int(robot_xy1[i][1]))
        donkey_frame = cv2.circle(donkey_frame, donkey_center, 3, (255, 0, 255), -1)                 # pink dot
        robot_frame = cv2.circle(robot_frame, robot_center1, 3, (255, 0, 255), -1)                   # pink dot
    return donkey_frame, robot_frame, interval


if __name__ == "__main__":

    # Connect to the robot
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP_ADDRESS, 5000))
    print('Connected')

    # Create a video capture object to read video stream from camera
    stream_cap = cv2.VideoCapture(STREAM_URL)

    found_donkey = False

    while stream_cap.isOpened():
        ret, robot_frame = stream_cap.read()
        if not ret: break

    ##### Robot Search #####################################################################################################
    
        # Detect keypoints and compute descriptors in the captured frame
        ret_keypoints, ret_descriptors, robot_frame = util.find_kpt_des(robot_frame, draw=True)

        # Set the robot's keypoints and descriptors to the captured frame's
        robot_keypoints = util.convert_keypoints(ret_keypoints)
        robot_descriptors = ret_descriptors

    ##### Donkey Search ####################################################################################################
    
        if not found_donkey:
            donkey_frame, robot_frame, interval = find_donkey(robot_descriptors, robot_keypoints, robot_frame)
            found_donkey = True

    ##### Carrot Persue ####################################################################################################
        
        # Load goal image keypoints (Carrot)
        goal_keypoints, goal_descriptors = util.load_descriptors(DES_FILE + "/" + DES_FILE + str((interval)+SKIP_INTERVAL) + ".yml")

        # Find the best matching keypoints between the robot and goal image
        rob_goal_matches = bf.match(robot_descriptors, goal_descriptors)
        rob_goal_matches = [m for m in rob_goal_matches if m.distance < MAX_MATCH_DISTANCE]
        print("# Goal kpt matches: " + str(len(rob_goal_matches)))

        if len(rob_goal_matches) < 3:
            print("No matches found")
            continue

        # Find matching keypoints coordinates for retrived and reference keypoints
        robot_xy2, carrot_xy = util.keypoint_coordinate(rob_goal_matches, robot_keypoints, goal_keypoints)

        # Find the goal image
        carrot_frame = show_reference_demo(((interval)+SKIP_INTERVAL))
        if carrot_frame is None: continue

        # Draw the matching keypoints
        for i in range(len(rob_goal_matches)):
            carrot_center = (int(carrot_xy[i][0]), int(carrot_xy[i][1]))
            robot_center2 = (int(robot_xy2[i][0]), int(robot_xy2[i][1]))    
            carrot_frame = cv2.circle(carrot_frame, carrot_center, 3, (255, 0, 0), -1)                             # blue dot
            carrot_frame = cv2.circle(carrot_frame, robot_center2, 3, (0, 0, 255), -1)                             # red dot
            carrot_frame = cv2.arrowedLine(carrot_frame, carrot_center, robot_center2, (0, 255, 255), thickness=1) # yellow line
            robot_frame = cv2.circle(robot_frame, robot_center2, 3, (0, 0, 255), -1)                               # red dot
            robot_frame = cv2.circle(robot_frame, carrot_center, 3, (255, 0, 0), -1)                               # blue dot
            robot_frame = cv2.arrowedLine(robot_frame, robot_center2, carrot_center, (0, 255, 255), thickness=1)   # yellow line

    ##### Control Analysis ####################################################################################################
        
        q_mx, q_my, q_rx, q_ry, t_mx, t_my, t_rx, t_ry = util.pair_analysis(robot_xy2, carrot_xy)
        cx_diff = 0
        rx_ratio = 1
        cx_diff = (t_mx - q_mx)
        rx_ratio = t_rx / q_rx
        print("x_diff: " + str(cx_diff) + " rx_ratio: " + str(rx_ratio))

        # Draw the covariance analysis
        # robot_frame = cv2.ellipse(robot_frame, (q_mx, q_my), (q_rx, q_ry), 0, 0, 360, (255, 255, 255), 1)                                  # black ellipse

    ##### Robot Control ######################################################################################################

        if abs(cx_diff) < 10:
            command = 'CMD_MOTOR#00#00#00#00\n'
        else:
            v = -20 * cx_diff
            if rx_ratio < 1:
                w = 30 * (rx_ratio - 1)
            else:
                w = 30 * (rx_ratio + 1)
            u = np.array([v - w, v + w])
            for _ in u:
                if _ >= 0:
                    _ = min(_, 570)
                    _ = max(_, 520)
                else:
                    _ = max(_, -570)
                    _ = min(_, -520)
            command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(u[0], u[0], u[1], u[1])

        s.send(command.encode('utf-8'))
        print(command)

        # Display the demo frame
        cv2.imshow('Carrot', carrot_frame)
        # Display the robot frame
        cv2.imshow('Robot', robot_frame)
        # Display the demo frame
        cv2.imshow('Donkey', donkey_frame)
        
        # Wait for Esc key to stop
        if cv2.waitKey(1) == 27:
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            stream_cap.release()
            break