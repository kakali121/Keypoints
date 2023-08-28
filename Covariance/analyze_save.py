import cv2
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


VIDEO_FILE = 'demo.mp4' # video file
MAX_FRAMES = 300  # large numbers will cover the whole video
INTERVAL = 5 # 150 frames per inverval
MAX_MATCH_DISTANCE = 20  # match threshold

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create()
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


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
        path = nx.shortest_path(G, source=oldest_v, target=v)  # FIXME Switch to longest path
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
    # Create an empty graph
    G = nx.DiGraph()
    # Add the edges to the graph
    G.add_edges_from(edges)
    return G


def find_timelines_from_video(frame_des, T=INTERVAL):
    G = sequential_matches_graph(frame_des, T)
    # Print some information about the graph
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    # Find the long paths that pass a threshold
    long_paths = find_long_paths_T(G)
    # Save video as a mp4 file
    # save_kp_video(frames, frame_kps, long_paths)
    return long_paths


def save_kpt_des(keypoints, descriptors, filename):
    # Create a file storage object
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    # Descriptors ##############################################################
    # Write the descriptors to the file
    fs.write("descriptors", descriptors)
    # Keypoints ###############################################################
    # Convert keypoints to a numpy array
    keypoints_array = np.array([keypoint.pt for keypoint in keypoints])
    # Write the keypoints to the file
    fs.write("keypoints", keypoints_array)  
    # Release the file
    fs.release()


def extract_keypoints(video):
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture(video)
    # Extract all keypoints and descriptors by frame
    frame_kps, frame_des = [], []
    video_frames = []
    k = 0 # frame counter
    # Loop through the video frames
    while cap.isOpened() and k < MAX_FRAMES:
        # Read a frame from the video
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
            kp2, des2 = orb.detectAndCompute(gray, None) # find the keypoints and descriptors with ORB
            if des2 is None:
                print("No keypoints/descriptors in frame ", k)
                continue
            frame_kps.append(kp2) # add keypoints to list
            frame_des.append(des2) # add descriptors to list
            video_frames.append(frame) # add frame to list
            k += 1
        else: break
    print("k =", k)
    cap.release()
    return frame_kps, frame_des, video_frames


def analyze_kpt_des(frame, keypoints, descriptors, filename, video):
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10 # frames per second
    out = cv2.VideoWriter(video, fourcc, fps, (400, 300)) # (width, height)
    # Create a list of current frame keypoints
    kpts_cur, des_cur = [], []
    # Create a list of reference frame keypoints
    kpts_fin, des_fin = [], []
    k = INTERVAL # set k to length of interval
    # Loop through the frames
    while k > 0:
        k -= 1
        if k == INTERVAL - 1:
            kpts_cur = keypoints[k] #current frame keypoints
            des_cur = descriptors[k] #current frame descriptors
            kpts_fin = keypoints[k] #reference frame keypoints
            des_fin = descriptors[k] #reference frame descriptors
            continue
        else:
            kpts_cur = keypoints[k] #current frame keypoints
            des_cur = descriptors[k] #current frame descriptors
        
        if descriptors is None:
            print("No keypoints/descriptors in frame")
            continue

        matches = bf.match(des_fin, descriptors[k])
        # matches = sorted(matches, key=lambda x: x.distance)
        matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]

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

        frame[k] = cv2.drawKeypoints(frame[k], kpts_cur, None, color = (255, 0, 0), flags=0)
        frame[k] = cv2.drawKeypoints(frame[k], kpts_fin, None, color=(0, 255, 0), flags=0)

        pt1s = []
        pt2s = []

        for i in range(len(kpts_fin)):
            pt1 = np.int32(kpts_fin[i].pt)
            pt2 = np.int32(kpts_cur[i].pt)
            pt1s.append(pt1)
            pt2s.append(pt2)
            frame[k] = cv2.line(frame[k], pt1, pt2, (0, 0, 255), thickness=2)

        # if len(pt1s) > 2 and len(pt2s) > 2:
        #     # Compute the convex hulls
        #     hull1 = ConvexHull(np.array(pt1s))
        #     hull2 = ConvexHull(np.array(pt2s))
        #     # Convert hull points to the correct format for cv2.drawContours()
        #     hull1_points = np.array(pt1s)[hull1.vertices]
        #     hull2_points = np.array(pt2s)[hull2.vertices]
        #     # Draw the contours on the frame
        #     cv2.drawContours(frame[k], [hull1_points], -1, (255, 255, 0), 3) # color blue
        #     cv2.drawContours(frame[k], [hull2_points], -1, (255, 0, 255), 3) # color pink

        cv2.imshow("frame", frame[k])
        out.write(frame[k])

        if k == 0:
            save_kpt_des(kpts_fin, des_fin, filename)
        # Wait for Esc key to stop
        if cv2.waitKey(1) == 27:
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            break



def time_descriptor(path, frame_kpts, frame_des):
    start = path[0][0]
    end = path[-1][0]
    #TODO include x and y
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
        if t >= time_stop: break
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
    color_paths= [tuple(map(int, np.random.randint(0, 255, size=3))) for _ in kp_paths]
    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width  = len(frames[0][0])
    height = len(frames[0])
    # print(width, height)
    fps = 10
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    for t, frame in enumerate(frames):
        for i, (start, end) in enumerate(intervals):
            if not (start <= t <= end):
                continue
            path = kp_paths[i]
            x, y = kp_path(frame_kps, path, t)
            cv_path = np.array([point for point in zip(x,y)], dtype=np.int32)
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


def plot_kp_timeline(paths):
    intervals = [(path[0][0], path[-1][0]) for path in paths]
    for i, (start, end) in enumerate(intervals):
        plt.plot([start, end], [i, i])
    plt.title("Keypoint timeline")
    plt.xlabel("Frame number")
    plt.ylabel("Keypoint id")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # ## Extract keypoints and descriptors
    frame_kpt, frame_des, frames = extract_keypoints(VIDEO_FILE)
    for i in range(int(len(frames)/INTERVAL)):
        print("Interval", i)
        analyze_kpt_des(frames, frame_kpt, frame_des, "demo_kpt_des/demo_kpt_des%d.yml"%(i+1), "demo_kpt_video/demo_kpt_video%d.mp4"%(i+1))
        frames = frames[-(len(frames)-INTERVAL):]
        frame_kpt = frame_kpt[-(len(frame_kpt)-INTERVAL):]
        frame_des = frame_des[-(len(frame_des)-INTERVAL):]