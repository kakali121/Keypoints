import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt


IMAGE_FILE = 'image.jpg' # image file
VIDEO_FILE = 'demo.mp4' # video file
INTERVAL = 5 # 150 frames per inverval
MAX_MATCH_DISTANCE = 60  # match threshold

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create()
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def keypoints_from_image_file(image_file):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, des = orb.detectAndCompute(gray, None)
    return keypoints, des, img


def load_descriptors(file_name):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    descriptors = fs.getNode("descriptors").mat()
    keypoints = fs.getNode("keypoints").mat()
    fs.release()
    return keypoints, descriptors


def draw_dist_interval(sum_distances):
    intervals = []
    for i in range(len([entry for entry in os.listdir("demo_kpt_des")])):
        intervals.append((i*INTERVAL, (i+1)*INTERVAL))
    x = np.hstack([(s, e) for (s, e) in intervals])
    y = np.hstack([(dist, dist) for dist in sum_distances])
    plt.plot(x, y, '-')
    for (s, e), dist in zip(intervals, sum_distances):
        x = [s,e]
        y = [dist, dist]
        plt.plot(x,y)
    # plt.plot(sum_distances)
    plt.title("Average distance of descriptors in interval")
    plt.show()


def draw_match(best_interval, img_keypoints):
    # Load the image
    img = cv2.imread(IMAGE_FILE)
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture(VIDEO_FILE)
    # Set the video's current frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, (best_interval+1)*INTERVAL)
    ret, bframe = cap.read()
    best_kpts, best_des = load_descriptors("demo_kpt_des/demo_kpt_des" + str(best_interval+1) + ".yml")
    best_match = bf.match(img_descriptors, best_des)
    # Plot best
    plt.imshow(bframe)
    for match in best_match:
        # Get the keypoints from the matches
        img1_idx = match.queryIdx # queryIdx is the index of the retrived keypoint 
        img2_idx = match.trainIdx # trainIdx is the index of the reference keypoint
        (x1, y1) = img_keypoints[img1_idx].pt
        (x2, y2) = best_kpts[img2_idx]
        # Draw a line between the keypoints with thicker line width
        plt.plot([x1, x2 + img.shape[1]], [y1, y2], linewidth=2, alpha=0.8)

    # Display the frame with matches
    # cv2.imshow('Frame with matches', frame_matches)
    #
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load the ORB feature detector and descriptor from the image
    img_keypoints, img_descriptors, img = keypoints_from_image_file(IMAGE_FILE)

    # find the interval that matches the most with the image descriptors
    sum_distances = []
    best_interval = -1
    best_sq_dist = math.inf

    for i in range(len([entry for entry in os.listdir("demo_kpt_des")])):
        # load the descriptor for the interval
        keypoints, descriptors = load_descriptors("demo_kpt_des/demo_kpt_des" + str(i+1) + ".yml")
        # match the descriptors with the image
        matches = bf.match(img_descriptors, descriptors)
        # print(len(matches))
        sum_distance = sum([m.distance for m in matches]) / len(matches)
        sum_distances.append(sum_distance)
        if sum_distance <= best_sq_dist:
            best_match = matches
            best_sq_dist = sum_distance
            best_interval = i
            best_kpts = keypoints
            best_des = descriptors
            # print("len best match", len(best_match))

    print("best interval", best_interval*INTERVAL, "-", (best_interval+1)*INTERVAL)
    draw_dist_interval(sum_distances)
    # draw_match(best_interval, img_keypoints)



        

