import cv2 
import math
import time
import numpy as np
from scipy.spatial import ConvexHull
import socket

IP_ADDRESS = '192.168.0.204'

file = "demo_kpt_des1/demo_kpt_des4.yml"

MAX_MATCH_DISTANCE = 60  # match threshold

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create(100)
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def load_descriptors(file_name):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    descriptors = fs.getNode("descriptors").mat()
    keypoints = fs.getNode("keypoints").mat()
    fs.release()
    return keypoints, descriptors


def remove_dup_coord(points):
    new_points = []
    list_x = []
    list_y = []
    for point in points:
        if point[0] not in list_x and point[1] not in list_y:
            new_points.append(point)
            list_x.append(point[0])
            list_y.append(point[1])
    return new_points


if __name__ == "__main__":

    # Connect to the robot
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP_ADDRESS, 5000))
    print('Connected')

    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture("http://192.168.0.204:1234/stream.mjpg")

    area_ref = []
    area = []
    ppt = []

    # reference keypoints and descriptors
    ref_kpts, ref_des = load_descriptors(file)

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # current keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # print("Ref Descriptors:", len(ref_des))
        matches = bf.match(ref_des, descriptors)
        matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]
        print("Matches:", len(matches))

        if len(matches) < 3: 
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord('q'):
                # De-allocate any associated memory usage
                cv2.destroyAllWindows()
                cap.release()
                break
            continue

        kpt_match = []
        des_match = []
        kpt_ref = []
        des_ref = []
        dist = []
        x_diff = []

        for match in matches:
            query_idx = match.queryIdx
            kpt_ref.append(ref_kpts[query_idx])
            des_ref.append(ref_des[query_idx])
            train_idx = match.trainIdx
            kpt_match.append(keypoints[train_idx])
            des_match.append(descriptors[train_idx])
            x_diff.append(ref_kpts[query_idx][0] - keypoints[match.trainIdx].pt[0])

        x = np.mean(x_diff)

        kpt_ref = np.array(kpt_ref)
        des_ref = np.array(des_ref)
        kpt_match = np.array(kpt_match)
        des_match = np.array(des_match)

        # dist = np.linalg.norm(des_ref - des_match, axis=1)
        
        frame = cv2.drawKeypoints(frame, kpt_match, None, color=(0, 255, 0), flags=0)
        # frame = cv2.drawKeypoints(frame, kpt_ref, None, color=(255, 0, 0), flags=0)

        pt1s = []
        pt2s = []

        for i in range(len(kpt_ref)):
            center = np.int32([kpt_ref[i][0], kpt_ref[i][1]])
            pt2 = np.int32(kpt_match[i].pt)
            if center[0] != pt2[0] or center[1] != pt2[1]:
                pt1s.append(center)
                pt2s.append(pt2)
                frame = cv2.circle(frame, center, 4, (255, 0, 0), -1) # color blue
                frame = cv2.line(frame, pt2, center, (0, 0, 255), thickness=2) # color red

        pt1s = remove_dup_coord(pt1s)
        pt2s = remove_dup_coord(pt2s)

        if len(pt1s) < 3: 
            command = 'CMD_MOTOR#00#00#00#00\n'
            s.send(command.encode('utf-8'))
            continue
        if len(pt2s) < 3: 
            command = 'CMD_MOTOR#00#00#00#00\n'
            s.send(command.encode('utf-8'))
            continue

        # Compute the convex hulls
        hull1 = ConvexHull(pt1s)
        hull2 = ConvexHull(pt2s)

        # Convert hull points to the correct format for cv2.drawContours()
        hull1_points = np.array(pt1s)[hull1.vertices]
        hull2_points = np.array(pt2s)[hull2.vertices]

        # Draw the contours on the frame
        cv2.drawContours(frame, [hull1_points], -1, (255, 255, 0), 3) # color green blue
        cv2.drawContours(frame, [hull2_points], -1, (255, 0, 255), 3) # color pink
        
        if len(area_ref) == 10:
            area_ref.pop(0)
            area.pop(0)
            ppt.pop(0)

        area_ref.append(hull1.area)
        area.append(hull2.area)
        ppt.append((hull1.area-hull2.area)/hull1.area)
        print("Reference:", np.mean(area_ref))
        print("Current:", np.mean(area))
        print("PPT:", np.mean(ppt))

        if len(ppt) < 10: continue

        cv2.imshow("frame", frame)
        
        v = 23 * -x
        # v = 700 if v > 0 else -700
        if (np.mean(ppt) > 0):
            w = 4500 * (np.mean(ppt) - 0.1)
        else:
            w = -4500 * (np.mean(ppt) - 0.1)

        u = np.array([v - w, v + w])
        u[u > 700.] = 700.
        u[u < -700.] = -700.

        command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(u[0], u[0], u[1], u[1])
        s.send(command.encode('utf-8'))
        print(command)

        # Wait for Esc key to stop
        if cv2.waitKey(1) == ord('q'):
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            command = 'CMD_MOTOR#00#00#00#00\n'
            s.send(command.encode('utf-8'))
            break

    command = 'CMD_MOTOR#00#00#00#00\n'
    s.send(command.encode('utf-8'))