'''
Author       : Karen Li
Date         : 2023-08-08 16:09:16
LastEditors  : Karen Li
LastEditTime : 2023-08-09 17:39:08
FilePath     : /WallFollowing/util.py
Description  : 
'''

import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create()
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def find_kpt_des(img, draw=False):
    '''
    description: 
    param       {Any} img: 
    param       {bool} draw: 
    return      {*}
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kpts, des = orb.detectAndCompute(gray, None)
    if draw: 
        img = cv2.drawKeypoints(img, kpts, None, color=(0, 255, 0), 
                                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # Green keypoints
    return kpts, des, img   


def convert_keypoints(keypoints)->np.ndarray:
    '''
    description: 
    param       {Any} keypoints: 
    return      {np.ndarray} points
    '''
    points = []
    for i in range(len(keypoints)):
        (x, y) = keypoints[i].pt
        points.append(np.array((x, y)))
    points = np.array(points)
    return points


def keypoints_from_image_file(image_file: str):
    '''
    description: 
    param       {str} image_file: 
    return      {*} keypoints, descriptors, image
    '''
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and compute descriptors in the frame
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors, image


def load_descriptors(file_name: str):
    '''
    description: 
    param       {str} file_name: 
    return      {*} keypoints, descriptors
    '''
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    descriptors = fs.getNode("descriptors").mat()
    keypoints = fs.getNode("keypoints").mat()
    fs.release()
    return keypoints, descriptors


def find_best_interval(img_descriptors, dir_name: str) -> int:
    '''
    description: 
    param       {*} img_descriptors: 
    param       {str} dir_name: 
    return      {*}
    '''
    ave_sum_distances = []
    best_interval = -1
    best_sq_dist = math.inf
    # Iterate through all intervals to find the interval that matches the most with the image descriptors
    for i in range(len([entry for entry in os.listdir(dir_name)])):
        # load the descriptor for the interval
        keypoints, descriptors = load_descriptors(
            dir_name + "/" + dir_name + str(i + 1) + ".yml"
        )
        # match the descriptors with the image
        matches = bf.match(img_descriptors, descriptors)
        if len(matches) == 0:
            continue
        # cummulate the average hamming distance for each interval match
        ave_sum_distance = sum([m.distance for m in matches]) / len(matches)
        ave_sum_distances.append(ave_sum_distance)
        # find the best match with the least distance
        if ave_sum_distance <= best_sq_dist:
            best_sq_dist = ave_sum_distance
            best_interval = i+1
    return best_interval


def keypoint_coordinate(matches, query_keypoints, train_keypoints):
    '''
    description: 
    param       {*} matches: 
    param       {*} query_keypoints: 
    param       {*} train_keypoints: 
    return      {*}
    '''
    query_xy, train_xy = [], []
    for match in matches:
        query_idx = match.queryIdx
        (x1, y1) = query_keypoints[query_idx]
        query_xy.append(np.array((x1, y1)))
        train_idx = match.trainIdx
        (x2, y2) = train_keypoints[train_idx]
        train_xy.append(np.array((x2, y2)))
    return query_xy, train_xy


def draw_confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    '''
    description: 
    param       {*} x: 
    param       {*} y: 
    param       {*} ax: 
    param       {*} n_std: 
    param       {*} facecolor: 
    param       {object} kwargs: 
    return      {*}
    '''
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Obtain the eigenvalues of this 2D dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )
    # Calculating the standard deviation of x
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    # calculating the standard deviation of y
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    # plot the center of the ellipse
    ax.plot(mean_x, mean_y, "r.")
    transf = transforms.Affine2D().scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    return (mean_x, mean_y, ell_radius_x, ell_radius_y)


def compute_confidence_ellipse(x, y, n_std=3.0, **kwargs):
    '''
    description: 
    param       {*} x: 
    param       {*} y: 
    param       {*} n_std: 
    param       {object} kwargs: 
    return      {*}
    '''
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Obtain the eigenvalues of this 2D dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    # Calculating the standard deviation of x
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    # calculating the standard deviation of y
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    return (mean_x, mean_y, ell_radius_x, ell_radius_y)


def plot_pair_analysis(query, query_name, train, train_name, h, w):
    '''
    description: 
    param       {*} query: 
    param       {*} query_name: 
    param       {*} train: 
    param       {*} train_name: 
    param       {*} h: 
    param       {*} w: 
    return      {*}
    '''
    # Keypoints of the obtained image
    qx = np.array([x for (x, y) in query])
    qy = np.array([y for (x, y) in query])
    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    plt.plot(qx, qy, ".")
    plt.xlim([0, w])
    plt.ylim([0, h])
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    q_mx, q_my, q_rx, q_ry = draw_confidence_ellipse(qx, qy, ax, edgecolor="red")
    plt.title(query_name + " Image Kpts")
    # Keypoints of the reference image
    tx = np.array([x for (x, y) in train])
    ty = np.array([y for (x, y) in train])
    plt.subplot(1, 2, 2)  # row 1, col 2 index 2
    plt.plot(tx, ty, ".")
    plt.xlim([0, w])
    plt.ylim([0, h])
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    t_mx, t_my, t_rx, t_ry = draw_confidence_ellipse(tx, ty, ax, edgecolor="red")
    plt.title(train_name + " Image Kpts")
    plt.show()
    print("cx deviation:", (q_mx - t_mx) / w)  # "how much away"
    print("x distribution ratio:", (q_rx / t_rx))  # "how close"


def pair_analysis(query, query_name, train, train_name, h, w):
    '''
    description: 
    param       {*} query: 
    param       {*} query_name: 
    param       {*} train: 
    param       {*} train_name: 
    param       {*} h: 
    param       {*} w: 
    return      {*}
    '''
    # Keypoints of the obtained image
    qx = np.array([x for (x, y) in query])
    qy = np.array([y for (x, y) in query])
    q_mx, q_my, q_rx, q_ry = compute_confidence_ellipse(qx, qy, edgecolor="red")
    # Keypoints of the reference image
    tx = np.array([x for (x, y) in train])
    ty = np.array([y for (x, y) in train])
    t_mx, t_my, t_rx, t_ry = compute_confidence_ellipse(tx, ty, edgecolor="red")
    print("cx deviation:", (q_mx - t_mx) / w)  # "how much away"
    print("x distribution ratio:", (q_rx / t_rx))  # "how close"
