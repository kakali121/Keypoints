import cv2 
import math
import time
import numpy as np
from cv2 import norm
# import networkx as nx
# import matplotlib.pyplot as plt

# WINDOW_T = 20
# VIDEO_FILE = '1.mp4'
# IMAGE_FILE = '1.jpg'
# MAX_FRAMES = 500  # large numbers will cover the whole video
# SHORTEST_LENGTH = 5  # min 5
MAX_MATCH_DISTANCE = 80  # match threshold

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create()
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def save_descriptors(keypoints, descriptors, file_name):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    # Write the descriptors to the file
    fs.write("descriptors", descriptors)
    # Convert keypoints to a list of dictionaries
    # keypoints_data = []
    # for keypoint in keypoints:
    #     keypoints_data.append({
    #         'pt': (keypoint.pt[0], keypoint.pt[1]),
    #         'size': keypoint.size,
    #         'angle': keypoint.angle,
    #         'response': keypoint.response,
    #         'octave': keypoint.octave,
    #         'class_id': keypoint.class_id
    #     })
    # # Write the keypoints to the file
    # fs.write("keypoints", keypoints_data)
    # Convert keypoints to a numpy array
    keypoints_array = np.array([keypoint.pt for keypoint in keypoints])
    # Write the keypoints to the file
    fs.write("keypoints", keypoints_array)  
    fs.release()


if __name__ == "__main__":
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture("http://192.168.0.204:1234/stream.mjpg")

    k = 0

    kpts_cur, des_cur = [], []
    kpts_fin, des_fin = [], []

    frame_kps, frame_des = [], []
    video_frames = []

    output_file = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (640, 480))

    while cap.isOpened() and k < 300:
        # Read a frame from the video
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        frame_kps.append(keypoints)
        frame_des.append(descriptors)
        print("Frame:", k, "Keypoints:", len(keypoints))

        if k == 0:
            kpts_cur = keypoints
            des_cur = descriptors
            kpts_fin = keypoints
            des_fin = descriptors
            k += 1
            continue
        else:
            kpts_cur = keypoints
            des_cur = descriptors

        if descriptors is None:
            print("No keypoints/descriptors in frame")
            continue

        matches = bf.match(des_fin, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]

        kpts_cur_temp = []
        des_cur_temp = []
        kpts_fin_temp = []
        des_fin_temp = []

        for match in matches:
            query_idx = match.queryIdx
            kpts_fin_temp.append(kpts_fin[query_idx])
            des_fin_temp.append(des_fin[query_idx])
            train_idx = match.trainIdx
            kpts_cur_temp.append(kpts_cur[train_idx])
            des_cur_temp.append(des_cur[train_idx])

        kpts_fin, des_fin = [], []
        kpts_fin = np.array(kpts_fin_temp)
        des_fin = np.array(des_fin_temp)
        kpts_cur, des_cur = [], []
        kpts_cur = np.array(kpts_cur_temp)
        des_cur = np.array(des_cur_temp)
        print("Final Keypoints:", len(kpts_fin))           

        # matches = sorted(matches, key=lambda x: x.distance)
        # print("Matches:", len(matches))

        frame = cv2.drawKeypoints(frame, kpts_cur, None, color = (255, 0, 0), flags=0)
        frame = cv2.drawKeypoints(frame, kpts_fin, None, color=(0, 255, 0), flags=0)

        k += 1

        # if k == 300:
        for i in range(len(kpts_fin)):
            pt1 = np.int32(kpts_fin[i].pt)
            pt2 = np.int32(kpts_cur[i].pt)
            frame = cv2.line(frame, pt1, pt2, (0, 0, 255), thickness=2)

        if k == 300:
            save_descriptors(kpts_fin, des_fin, "kpts_des.yml")

        # cv2.imshow("frame", frame)
        out.write(frame)

        # Wait for Esc key to stop
        if cv2.waitKey(33) == 27:
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            break

    out.release()
