import cv2 
import math
import time
import numpy as np
import socket

IP_ADDRESS = '192.168.0.204'

# Connect to the robot
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((IP_ADDRESS, 5000))
print('Connected')

MAX_MATCH_DISTANCE = 30  # match threshold
POSITIONS = 4 # number of positions to go to

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

    while cap.isOpened() and position < POSITIONS + 1:
        # Read a frame from the video
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        ref_kpts, ref_des = load_kpt_des("demo_kpt_des/demo_kpt_des%d.yml"%(position))
        # print("Ref Descriptors:", len(ref_des))
        matches = bf.match(ref_des, descriptors)
        matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]
        print("Going to position:", position)
        print("Matches:", len(matches))

        if len(matches) < 1: continue

        kpt_match = []
        des_match = []
        kpt_ref = []
        des_ref = []
        x_diff = []

        for match in matches:
            query_idx = match.queryIdx
            kpt_ref.append(ref_kpts[query_idx])
            des_ref.append(ref_des[query_idx])
            train_idx = match.trainIdx
            kpt_match.append(keypoints[train_idx])
            des_match.append(descriptors[train_idx])
            x_diff.append(ref_kpts[query_idx][0] - keypoints[match.trainIdx].pt[0])

        matches = sorted(matches, key=lambda x: x.distance)
        n = int(0.7*len(matches))
        matches = matches[:n]
        
        x_diff = []

        for match in matches:
            x_diff.append(ref_kpts[match.queryIdx][0] - keypoints[match.trainIdx].pt[0])

        x = np.mean(x_diff)
        print("Average x diff:", np.mean(x_diff))
        
        if math.isnan(x):
            print("No matches")
            continue
        else:
            if abs(x) > 6:
                v = 49 * -x
                w = 0
                u = np.array([v - w, v + w])
                u[u > 600.] = 600.
                u[u < -600.] = -600.
                command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(u[0], u[0], u[1], u[1])
                print(command)
                s.send(command.encode('utf-8'))
                print(command)
            else:
                position += 1
                command = 'CMD_MOTOR#00#00#00#00\n'
                s.send(command.encode('utf-8'))
                if position > POSITIONS:
                    print("Done")
                    break
                print("Going to position:", position)

        kpt_ref = np.array(kpt_ref)
        des_ref = np.array(des_ref)
        kpt_match = np.array(kpt_match)
        des_match = np.array(des_match)
        
        frame = cv2.drawKeypoints(frame, kpt_match, None, color=(0, 255, 0), flags=0)
        # frame = cv2.drawKeypoints(frame, kpt_ref, None, color=(255, 0, 0), flags=0)

        for i in range(len(kpt_ref)):
            center = int(kpt_ref[i][0]), int(kpt_ref[i][1])
            # print(center)
            frame = cv2.circle(frame, center, 4, (255, 0, 0), -1)
            # frame = cv2.line(frame, center, kpt_match, (0, 0, 255), thickness=2)
            pt1 = np.int32(kpt_match[i].pt)
            # pt2 = np.int32(kpt_ref[i].pt)
            frame = cv2.line(frame, pt1, center, (0, 0, 255), thickness=2)

        cv2.imshow("frame", frame)

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
        
