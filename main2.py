import cv2 
import math
import time
import numpy as np
from cv2 import norm

MAX_MATCH_DISTANCE = 80  # match threshold

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create()
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def load_descriptors(file_name):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    descriptors = fs.getNode("descriptors").mat()
    keypoints = fs.getNode("keypoints").mat()
    fs.release()
    return keypoints, descriptors

if __name__ == "__main__":
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture("http://192.168.0.204:1234/stream.mjpg")

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        # frame_kps.append(keypoints)
        # frame_des.append(descriptors)
        # print("Frame:", k, "Keypoints:", len(keypoints))
        ref_kpts, ref_des = load_descriptors("kpts_des.yml")
        # print("Ref Descriptors:", len(ref_des))
        matches = bf.match(ref_des, descriptors)
        matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]
        print("Matches:", len(matches))

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

        kpt_ref = np.array(kpt_ref)
        des_ref = np.array(des_ref)
        kpt_match = np.array(kpt_match)
        des_match = np.array(des_match)
        
        frame = cv2.drawKeypoints(frame, kpt_match, None, color=(0, 255, 0), flags=0)
        # frame = cv2.drawKeypoints(frame, kpt_ref, None, color=(255, 0, 0), flags=0)

        for i in range(len(kpt_ref)):
            print(kpt_ref[i])
            center = int(kpt_ref[i][0]), int(kpt_ref[i][1])
            # print(center)
            frame = cv2.circle(frame, center, 2, (255, 0, 0), -1)
            # frame = cv2.line(frame, center, kpt_match, (0, 0, 255), thickness=2)
            pt1 = np.int32(kpt_match[i].pt)
            # pt2 = np.int32(kpt_ref[i].pt)
            frame = cv2.line(frame, pt1, center, (0, 0, 255), thickness=2)

        cv2.imshow("frame", frame)

        # Wait for Esc key to stop
        if cv2.waitKey(33) == 27:
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            break