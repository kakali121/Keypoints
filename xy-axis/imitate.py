import cv2 
import os
import math
import time
import numpy as np
from scipy.spatial import ConvexHull
import socket

IP_ADDRESS = '192.168.0.204'

DES_DIR = "demo_kpt_des2/"
DES_FILE = "demo_kpt_des"

STREAM_URL = "http://192.168.0.204:1234/stream.mjpg"
DEMO_VIDEO = "demo2.mp4"

SKIP_INTERVAL = 3
MAX_MATCH_DISTANCE = 50  # match threshold

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create(500)
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def get_total_interval():
    total = 0
    for f in os.listdir(DES_FILE):
        if f == ".DS_Store": continue
        total += 1
    return total


def load_kpt_des(file_name):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    keypoints = fs.getNode("keypoints").mat()
    descriptors = fs.getNode("descriptors").mat()
    fs.release()
    return keypoints, descriptors


def find_kpt_des(img, draw=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kpts, des = orb.detectAndCompute(gray, None)
    if draw: img = cv2.drawKeypoints(img, kpts, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # Red keypoints
    return kpts, des, img   


def find_best_reference(ret_des):
    ave_sum_distances = []
    best_interval = -1
    best_sq_dist = math.inf
    # Iterate through all intervals to find the interval that matches the most with the image descriptors
    for i in range(len(os.listdir(DES_FILE))):
        # load the descriptor for the interval
        ref_kpt, ref_des = load_kpt_des(DES_DIR + DES_FILE + str(i+1) + ".yml")
        # match the descriptors with the image
        matches = bf.match(ret_des, ref_des)
        matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]
        if len(matches) < 1: continue
        # cummulate the average hamming distance for each interval match
        ave_sum_distance = sum([m.distance for m in matches]) / len(matches)
        ave_sum_distances.append(ave_sum_distance)
        # find the best match with the least distance
        if ave_sum_distance <= best_sq_dist:
            best_sq_dist = ave_sum_distance
            best_interval = i + 1
    return best_interval


def get_ref_ret_kpt(matches, ref_keypoints, ref_descriptors, ret_keypoints, ret_descriptors, img, draw=False):
    kpt_ref, des_ref, kpt_ret, des_ret = [], [], [], []
    for match in matches:
        query_idx = match.queryIdx
        kpt_ref.append(ref_keypoints[query_idx])
        des_ref.append(ref_descriptors[query_idx])
        train_idx = match.trainIdx
        kpt_ret.append(ret_keypoints[train_idx])
        des_ret.append(ret_descriptors[train_idx])
    kpt_ref = np.array(kpt_ref)
    des_ref = np.array(des_ref)
    kpt_ret = np.array(kpt_ret)
    des_ret = np.array(des_ret)
    if draw: img = cv2.drawKeypoints(frame, kpt_ret, None, color=(0, 255, 0), flags=0) # Green all robot keypoints
    return kpt_ref, des_ref, kpt_ret, des_ret, img


def get_ref_ret_coordinates(kpt_ref, kpt_ret, img, draw=False):
    pt1s, pt2s = [], []
    for i in range(len(kpt_ref)):
        pt1 = np.int32([kpt_ref[i][0], kpt_ref[i][1]])
        pt1s.append(pt1)
        pt2 = np.int32(kpt_ret[i].pt)
        pt2s.append(pt2)
        if draw:
            img = cv2.circle(img, pt1, 4, (255, 0, 0), -1) # blue matched reference keypoints
            img = cv2.line(img, pt 2, pt1, (255, 255, 0), thickness=2) # red matched robot keypoints
    pt1s = np.array(pt1s)
    pt2s = np.array(pt2s)
    return pt1s, pt2s, img


def confidence_x(x, y, n_std=3.0):
    if x.size != y.size: raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    radius_x = np.sqrt(1 + pearson)
    width = radius_x * 2
    scale_x = np.sqrt(cov[0, 0]) * n_std
    width_x = width * scale_x
    mean_x = np.mean(x)
    return mean_x, width_x   


if __name__ == "__main__":
    # Connect to the robot
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP_ADDRESS, 5000))
    print('Connected')
    # Create a VideoCapture object to read the video file
    # cap = cv2.VideoCapture("demo2.mp4")
    cap = cv2.VideoCapture("http://192.168.0.204:1234/stream.mjpg")

    ref_interval = -1
    togo_interval = -1
    total = get_total_interval()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Find keypoints from the captured frame
        ret_keypoints, ret_descriptors, frame = find_kpt_des(frame, draw=True)
        # Find the best matching interval reference
        ref_interval = find_best_reference(ret_descriptors)
        if ref_interval == -1: 
            print("No matching interval found")
            continue
        # Load keypoints from the interval reference
        ref_keypoints, ref_descriptors = load_kpt_des(DES_DIR + DES_FILE + str(ref_interval+1) + ".yml")
        # Match the captured against the reference
        matches = bf.match(ref_descriptors, ret_descriptors)

        print("Matching interval: " + str(ref_interval+1))
        print("Keypoint matches:", len(matches))

        if len(matches) < 3: continue

        kpt_ref, des_ref, kpt_ret, des_ret, frame = get_ref_ret_kpt(matches, ref_keypoints, ref_descriptors, ret_keypoints, ret_descriptors, frame, draw=True)
        ref_pts, ret_pts, frame = get_ref_ret_coordinates(kpt_ref, kpt_ret, frame, draw=True)

        ref_mean_x, ref_width_x = confidence_x(ref_pts[:, 0], ref_pts[:, 1])
        ret_mean_x, ret_width_x = confidence_x(ret_pts[:, 0], ret_pts[:, 1])

        print("Axis ratio", ref_width_x/ret_width_x)
        print("Center deviation", ref_mean_x-ret_mean_x)

        cv2.imshow("frame", frame)

        v = 33 * -(ref_mean_x-ret_mean_x)
        # v = 700 if v > 0 else -700
        if (ref_width_x/ret_width_x > 0): # too far
            w = 200 * (ref_width_x/ret_width_x)
        else: # too close
            w = -200 * (ref_width_x/ret_width_x)

        u = np.array([v - w, v + w])
        u[u > 700.] = 700.
        u[u < -700.] = -700.

        command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(u[0], u[0], u[1], u[1])
        s.send(command.encode('utf-8'))
        print(command)

        # Wait for Q key to stop
        if cv2.waitKey(1) == ord('q'):
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            command = 'CMD_MOTOR#00#00#00#00\n'
            s.send(command.encode('utf-8'))
            break

    command = 'CMD_MOTOR#00#00#00#00\n'
    s.send(command.encode('utf-8'))





# if len(matches) < 3: 
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) == ord('q'):
#         # De-allocate any associated memory usage
#         cv2.destroyAllWindows()
#         cap.release()
#         break
#     continue

# # Compute the convex hulls
# hull1 = ConvexHull(pt1s, qhull_options='QJ')
# hull2 = ConvexHull(pt2s, qhull_options='QJ')
# # Convert hull points to the correct format for cv2.drawContours()
# hull1_points = np.array(pt1s)[hull1.vertices]
# hull2_points = np.array(pt2s)[hull2.vertices]
# # Draw the contours on the frame
# cv2.drawContours(frame, [hull1_points], -1, (255, 255, 0), 3) # color green blue
# cv2.drawContours(frame, [hull2_points], -1, (255, 0, 255), 3) # color pink

# if len(area_ref) == 10:
#     area_ref.pop(0)
#     area.pop(0)
#     ppt.pop(0)

# area_ref.append(hull1.area)
# area.append(hull2.area)
# ppt.append((hull1.area-hull2.area)/hull1.area)

# print("Reference:", np.mean(area_ref))
# print("Current:", np.mean(area))
# print("PPT:", np.mean(ppt))

# if len(ppt) < 10: continue

# cv2.imshow("frame", frame)

# if abs(x) > 6:
#     v = 33 * -x
#     # v = 700 if v > 0 else -700
#     if (np.mean(ppt) > 0):
#         w = 1500 * (np.mean(ppt) - 0.1)
#     else:
#         w = -1500 * (np.mean(ppt) - 0.1)
#     u = np.array([v - w, v + w])
#     u[u > 700.] = 700.
#     u[u < -700.] = -700.
#     command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(u[0], u[0], u[1], u[1])
#     s.send(command.encode('utf-8'))
#     print(command)
# else:
#     current += 1
#     command = 'CMD_MOTOR#00#00#00#00\n'
#     s.send(command.encode('utf-8'))
#     if current <= ( len([entry for entry in os.listdir(DES_FILE)]) / SKIP_INTERVAL ):
#         command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(0, 0, 0, 0)
#         s.send(command.encode('utf-8'))
#         print(command)
#         break
#     print("Going to position:", current)