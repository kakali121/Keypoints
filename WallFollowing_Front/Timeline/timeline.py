import cv2
import math
import time
import numpy as np
from cv2 import norm
import networkx as nx
import matplotlib.pyplot as plt

WINDOW_T = 20
VIDEO_FILE = 'frontdemo.mp4'
# IMAGE_FILE = '1.jpg'
MAX_FRAMES = 2400                       # large numbers will cover the whole video
SHORTEST_LENGTH = 5                     # min 5
MAX_MATCH_DISTANCE = 40                 # match threshold

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create()
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def keypoints_from_image_file(image_file):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and compute descriptors in the frame
    keypoints, des = orb.detectAndCompute(gray, None)
    # print(keypoints)
    # img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display the image with keypoints
    # plt.imshow(img_with_keypoints)
    # plt.show()
    return keypoints, des, img


def find_times(time_kps):
    times = []
    for s, e, kpts, dess in time_kps:
        if not s in times:
            times.append(s)
        if not e in times:
            times.append(e)
    return sorted(times)


def descriptors_per_interval(times, time_kps, frame_kps, frame_des, plot=False):
    # For each interval in a sequence, find the keypoints
    intervals_in_sequence = []
    for s1, e1 in zip(times[:-1], times[1:]):
        intervals_in_sequence.append((s1, e1, frame_kps[s1], frame_des[s1]))
        # print(s1, e1, len(interval_descriptors))
    if plot:
        plt.plot([len(dess) for s, e, kpts, dess in intervals_in_sequence])
        plt.title("Number of Keypoints per interval")
        plt.show()
    return intervals_in_sequence


def find_long_paths_T(G):
    # Invert the graph
    G_inv = nx.DiGraph()
    G_inv.add_edges_from([(v, u) for u, v in G.edges])
    visited_flag = {v: 0 for v in G.nodes}
    paths = []
    # Run topological sort
    sorted_vertices = list(nx.topological_sort(G_inv))
    # for each subtree
    for v in sorted_vertices:
        print("v:", v)
        # v was already visited?
        if visited_flag[v]:
            continue
        v_decendents = nx.descendants(G_inv, v)
        oldest_t = math.inf
        oldest_v = None
        for u in v_decendents:
            # mark as visited
            visited_flag[u] = 1
            # find the oldest
            t, _ = u
            if t < oldest_t:
                oldest_t = t
                oldest_v = u
        if oldest_v is None:
            continue
        path = nx.shortest_path(
            G, source=oldest_v, target=v
        )  # FIXME Switch to longest path
        paths.append(path)
    # Find the paths with length greater than a threshhold. remove the source and sink nodes.
    long_paths = [p for p in paths if len(p) >= SHORTEST_LENGTH]
    print("Long paths:", len(long_paths))
    # sort by frame time
    long_paths = sorted(long_paths, key=lambda x: x[0])
    return long_paths


def find_best_matches(frame_des1, frame_des2):
    # matches(query_this_image, train)
    matches = bf.match(frame_des1, frame_des2)
    # Best matches: Filter by max distance
    matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]
    best_edges = [(m.queryIdx, m.trainIdx) for m in matches]
    return best_edges


def sequential_matches_graph(frame_des, T=1):
    edges = []  # ((time, kp_id), (time, kp_id))
    for i in range(len(frame_des) - T):
        for t in range(1, T + 1):
            # matches(query_this_image, train)
            matches = bf.match(frame_des[i], frame_des[i + t])
            # Best matches: Filter by max distance
            matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]
            best_edges = [(m.queryIdx, m.trainIdx) for m in matches]
            for e in best_edges:
                edges.append(((i, e[0]), (i + t, e[1])))
        if i%10 == 0:
            print("progress:", i, "of", len(frame_des) - T)
    # Create an empty graph
    G = nx.DiGraph()
    # Add the edges to the graph
    G.add_edges_from(edges)
    return G


def find_timelines_from_video(frame_des, T=WINDOW_T):
    G = sequential_matches_graph(frame_des, T)
    # Print some information about the graph
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    # Find the long paths that pass a threshold
    long_paths = find_long_paths_T(G)
    # Save video as a mp4 file
    # save_kp_video(frames, frame_kps, long_paths)
    return long_paths


def extract_keypoints(video, max_frames):
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture(video)
    # Extract all keypoints and descriptors by frame
    frame_kps, frame_des = [], []
    video_frames = []
    k = 0
    # Loop through the video frames
    while cap.isOpened() and k < max_frames:
        # Read a frame from the video
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(gray, None)
            if des2 is None:
                print("No keypoints/descriptors in frame ", k)
                continue
            frame_kps.append(kp2)
            frame_des.append(des2)
            video_frames.append(frame)
            k += 1
        else:
            break
    print("k =", k)
    cap.release()
    return frame_kps, frame_des, video_frames


def time_descriptor(path, frame_kpts, frame_des):
    start = path[0][0]
    end = path[-1][0]
    # TODO include x and y
    descriptor = None
    # first element in path
    t, kp_id = path[0]
    descriptor = frame_des[t][kp_id]
    kpts = frame_kpts[t][kp_id]
    return start, end, kpts, descriptor


def kp_path(frame_kps, path, time_stop=-1):
    x, y = [], []
    for vertex in path:
        t, kp_id = vertex
        kp = frame_kps[t][kp_id]
        # print(kp.pt)
        x.append(kp.pt[0])
        y.append(kp.pt[1])
        if t >= time_stop:
            break
    return x, y


def time_in_path(time, path, frame_kps):
    for vertex in path:
        t, kp_id = vertex
        if t >= time:
            kp = frame_kps[t][kp_id]
            # print(kp.pt)
            px = kp.pt[0]
            py = kp.pt[1]
            return px, py


def save_kp_video(frames, frame_kps, kp_paths):
    intervals = [(path[0][0], path[-1][0]) for path in kp_paths]
    color_paths = [tuple(map(int, np.random.randint(0, 255, size=3))) for _ in kp_paths]
    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = len(frames[0][0])
    height = len(frames[0])
    # print(width, height)
    fps = 20
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    for t, frame in enumerate(frames):
        for i, (start, end) in enumerate(intervals):
            if not (start <= t <= end):
                continue
            path = kp_paths[i]
            x, y = kp_path(frame_kps, path, t)
            cv_path = np.array([point for point in zip(x, y)], dtype=np.int32)
            # Generate a random color for the polyline
            color = color_paths[i]
            # Draw
            cv2.polylines(frame, [cv_path], isClosed=False, color=color, thickness=2)
            point = np.array(time_in_path(t, path, frame_kps), dtype=np.int32)
            cv2.circle(frame, point, 5, color, -1)
        # cv2.imshow('Video Frame', frame)
        out.write(frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        #
        # sleep(.1)
    out.release()


def plot_kp_timeline(paths: list, save_plot=True) -> None:
    intervals = [(path[0][0], path[-1][0]) for path in paths]
    for i, (start, end) in enumerate(intervals):
        plt.plot([start, end], [i, i])
    plt.title("Keypoint timeline")
    plt.xlabel("Frame number")
    plt.ylabel("Keypoint id")
    plt.grid()

    if save_plot:
        plt.savefig("timeline.png")
    else:
        plt.show()

if __name__ == "__main__":
    # ## Extract keypoints
    frame_kps, frame_des, frames = extract_keypoints(VIDEO_FILE, max_frames=MAX_FRAMES)

    # ## Find sequential match
    G = sequential_matches_graph(frame_des, WINDOW_T)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    # ## Find the long paths
    print("Finding long paths...")
    long_paths = find_long_paths_T(G)

    # ## Plot keypoint timeline
    print("Plotting keypoint timeline...")
    plot_kp_timeline(long_paths)

    # ## Save video with keypoints
    print("Saving video with keypoints...")
    save_kp_video(frames, frame_kps, long_paths)

    # ## Find descriptor - first frame
    time_kps = [time_descriptor(path, frame_kps, frame_des) for path in long_paths]

    # ## Make time interval
    times = find_times(time_kps)

    # ## Find interval descriptor
    intervals_in_sequence = descriptors_per_interval(
        times, time_kps, frame_kps, frame_des, plot=True
    )
    print("Number of intervals:", len(intervals_in_sequence))

    #### Find best match ################################################################################

    # # ## Read image
    # img_keypoints, img_descriptors, img = keypoints_from_image_file(IMAGE_FILE)
    # # img_keypoints, img_descriptors, img = frame_kps[20], frame_des[20], frames[20]

    # # find the interval that matches the most with the image descriptors
    # sum_distances = []
    # best_match = None
    # best_interval_id = -1
    # best_sq_dist = math.inf
    # best_kpts = None
    # best_des = None

    # for i, (s, e, kpts, descriptors) in enumerate(intervals_in_sequence):
    #     matches = bf.match(img_descriptors, descriptors)
    #     # matches = sorted(matches, key=lambda x: x.distance)
    #     # print(s, e, len(matches))
    #     sum_distance = sum([m.distance for m in matches]) / len(matches)
    #     sum_distances.append(sum_distance)
    #     if sum_distance <= best_sq_dist:
    #         best_match = matches
    #         best_sq_dist = sum_distance
    #         best_interval_id = i
    #         best_kpts = kpts
    #         best_des = descriptors
    #         # print("len best match", len(best_match))
    # # print(sum_distances)
    # x = np.hstack([(s, e) for (s, e, kpts, img_descriptors) in intervals_in_sequence])
    # y = np.hstack([(dist, dist) for dist in sum_distances])
    # plt.plot(x, y, '-')
    # for (s, e, kpts, img_descriptors), dist in zip(intervals_in_sequence, sum_distances):
    #     x = [s,e]
    #     # print(x)
    #     y = [dist, dist]
    #     plt.plot(x,y)
    # # plt.plot(sum_distances)
    # plt.title("Average distance of descriptors in interval")
    # plt.show()

    # # best interval
    # bs, be, bkpts, bdess = intervals_in_sequence[best_interval_id]

    # # best_frame = frames[bs]
    # # best_kpts = frame_kps[bs]
    # # bdess = frame_des[bs]
    # # best_kpts = frame_kps[bs]
    # # best_match = bf.match(img_descriptors, bdess)
    # print('Number of best matches', len(best_match), " time =", bs, "to", be)
    # # best_match = sorted(best_match, key=lambda x: x.distance)[:50]
    # # frame_matches = cv2.drawMatches(img, img_keypoints, best_frame, best_kpts, best_match, None,
    # #                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # # # Plot best
    # # plt.imshow(frame_matches)
    # # for match in best_match:
    # #     # Get the keypoints from the matches
    # #     img1_idx = match.queryIdx
    # #     img2_idx = match.trainIdx
    # #     (x1, y1) = img_keypoints[img1_idx].pt
    # #     (x2, y2) = best_kpts[img2_idx].pt

    # #     # Draw a line between the keypoints with thicker line width
    # #     plt.plot([x1, x2 + img.shape[1]], [y1, y2], linewidth=2, alpha=0.8)

    # # # Display the frame with matches
    # # # cv2.imshow('Frame with matches', frame_matches)
    # # #
    # # # cv2.waitKey(0)
    # # # cv2.destroyAllWindows()

    # # # break
    # # plt.show()
    # # print(len(time_kps))
