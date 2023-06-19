import cv2 
import math
import numpy as np
from scipy.spatial import ConvexHull
import socket

IP_ADDRESS = '192.168.0.204'

# Connect to the robot
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((IP_ADDRESS, 5000))
print('Connected')

MAX_MATCH_DISTANCE = 30  # match threshold
POSITIONS = 3 # number of positions to go to

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create(200)
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def load_kpt_des(file_name):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    descriptors = fs.getNode("descriptors").mat()
    keypoints = fs.getNode("keypoints").mat()
    fs.release()
    return keypoints, descriptors


if __name__ == "__main__":
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture("http://192.168.0.204:1234/stream.mjpg")
    position = 1

    area_ref = []
    area = []

    ref_kpts, ref_des = load_kpt_des("demo_kpt_des/demo_kpt_des%d.yml"%(position))

    while cap.isOpened() and position < POSITIONS + 1:
        print(len(ref_kpts))
        # Read a frame from the video
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if not ret: break
        # Increase the contrast
        # alpha = 2  # Contrast control (1.0 to 3.0, where higher values increase contrast)
        # frame = cv2.convertScaleAbs(frame, alpha=alpha)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 0, 255), flags=0) # color green
        # print("Ref Descriptors:", len(ref_des))
        matches = bf.match(ref_des, descriptors)
        # matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]

        print("Going to position:", position)
        print("Matches:", len(matches))

        if len(matches) < 3: continue

        kpt_match = []
        des_match = []
        kpt_ref = []
        des_ref = []

        for match in matches:
            query_idx = match.queryIdx
            kpt_ref.append(ref_kpts[query_idx])
            des_ref.append(ref_des[query_idx])
            train_idx = match.trainIdx
            kpt_match.append(keypoints[train_idx])
            des_match.append(descriptors[train_idx])

        # matches = sorted(matches, key=lambda x: x.distance)
        # n = int(0.7*len(matches))
        # matches = matches[:n]

        kpt_ref = np.array(kpt_ref)
        des_ref = np.array(des_ref)
        kpt_match = np.array(kpt_match)
        des_match = np.array(des_match)
        
        frame = cv2.drawKeypoints(frame, kpt_match, None, color=(0, 255, 0), flags=0) # color green
        # frame = cv2.drawKeypoints(frame, kpt_ref, None, color=(255, 0, 0), flags=0)

        pt1s = []
        pt2s = []

        for i in range(len(kpt_ref)):
            center = np.int32([kpt_ref[i][0], kpt_ref[i][1]])
            pt1s.append(center)
            # print(center)
            frame = cv2.circle(frame, center, 4, (255, 0, 0), -1) # color blue
            # frame = cv2.line(frame, center, kpt_match, (0, 0, 255), thickness=2)
            pt2 = np.int32(kpt_match[i].pt)
            pt2s.append(pt2)
            # pt2 = np.int32(kpt_ref[i].pt)
            frame = cv2.line(frame, pt2, center, (0, 0, 255), thickness=2) # color red

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
        area_ref.append(hull1.area)
        area.append(hull2.area)
        print("Reference:", np.mean(area_ref))
        print("Current:", np.mean(area))

        cv2.imshow("frame", frame)

        if len(area_ref) < 10: continue

        print((np.mean(area_ref) - np.mean(area))/np.mean(area_ref))
        
        if (np.mean(area_ref) - np.mean(area))/np.mean(area_ref) > 0.05:
            v = 6900 * (np.mean(area_ref) - np.mean(area))/np.mean(area_ref)
            w = 0
            u = np.array([v - w, v + w])
            u[u > 600.] = 600.
            u[u < -600.] = -600.
            command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(u[0], u[0], u[1], u[1])
            s.send(command.encode('utf-8'))
            print(command)
        else:
            position += 1
            if position > POSITIONS:
                print("Done")
                # break
            if position < POSITIONS + 1:
                ref_kpts, ref_des = load_kpt_des("demo_kpt_des/demo_kpt_des%d.yml"%(position))
            area_ref = []
            area = []
            # command = 'CMD_MOTOR#00#00#00#00\n'
            # s.send(command.encode('utf-8'))
            print("Going to position:", position)

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
        
