import matplotlib.pyplot as plt
import numpy as np
import cv2

### Constants ###
# Path to the descriptor files
DESCRIPTOR_FILE_PATH = "side_demo_kpt_des"      
# The maximum distance between two matched keypoints
MAX_MATCH_DISTANCE = 40             
# THe nummber of frames in an interval
FRAMES_PER_INTERVAL = 12
# Maximum number of intervals
MAX_INTERVALS = 300
# Create a BFMatcher object with Hamming distance (suitable for ORB, BRIEF, etc.)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)            


def load_kpt_des() -> list:
    """
    description: Load KPT descriptors from the files.
    return      {list}: A list containing the descriptors for each interval.
    """
    # List of all the descriptors
    descriptors = []
    # Load and save all the descriptors from the file
    for i in range(MAX_INTERVALS):
        file_name = (
            "../"
            + DESCRIPTOR_FILE_PATH
            + "/"
            + DESCRIPTOR_FILE_PATH
            + str(i + 1)
            + ".yml"
        )
        file_storage = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        # Load the descriptors from the file
        descriptors.append(file_storage.getNode("descriptors").mat())
        file_storage.release()
    return descriptors


def compare_descriptors(descriptors1: np.ndarray, descriptors2: np.ndarray) -> list:
    """
    description: Compare two sets of descriptors using BFMatcher.
    param       {np.ndarray} descriptors1: Descriptors from the first set.
    param       {np.ndarray} descriptors2: Descriptors from the second set.
    return      {list}: A list containing pairs of matching keypoints' indices (queryIdx, trainIdx).
    """
    # Match descriptors from two intervals
    raw_matches = bf.match(descriptors1, descriptors2)
    # Extract pairs of matching keypoints' indices from the matches
    raw_matches = [m for m in raw_matches if m.distance < MAX_MATCH_DISTANCE]
    matches = [(match.queryIdx, match.trainIdx) for match in raw_matches]
    return matches


def process_data(descriptors: np.ndarray) -> np.ndarray:
    """
    description: Process the data to get the matches between each pair of intervals.
    param       {np.ndarray} descriptors: DataFrame containing the descriptors for each interval.
    return      {np.ndarray}: A list containing the matches between each pair of intervals.
    """
    # List of all the matches
    all_matches = []
    # Extract pairs of matching keypoints' indices from the matches
    for i in range(len(descriptors) - 1):
        current_matches = compare_descriptors(descriptors[i], descriptors[i + 1])
        all_matches.append(current_matches)
    return all_matches


def compare_matches(all_matches: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
    """
    description: Compare the matches between each pair of intervals.
    param       {np.ndarray} all_matches: DataFrame containing the matches between each pair of intervals.
    return      {np.ndarray}: A list containing the matches between each pair of intervals.
    """
    # List to store the continuous matches and the terminal matches
    continues_keypoints = []
    terminated_keypoints = []
    # Extract pairs of matching keypoints' indices from the matches
    for i, matches in enumerate(all_matches):
        # Extract the indices of the matching keypoints
        match_pair_1, match_pair_2 = zip(*matches)
        # Extract the indices of the matching keypoints from the last interval
        last_matches = [x[1] for x in continues_keypoints]
        for j in range(len(descriptors[i])):
            if j in match_pair_1:
                if j in last_matches:
                    index = last_matches.index(j)
                    continues_keypoints[index][1] = match_pair_2[match_pair_1.index(j)]
                    continues_keypoints[index][3] = i + 2
                else:
                    continues_keypoints.append(
                        [j, match_pair_2[match_pair_1.index(j)], i + 1, i + 2]
                    )
            else:
                if j in last_matches:
                    terminated_keypoints.append(
                        continues_keypoints[last_matches.index(j)]
                    )
                else:
                    terminated_keypoints.append([j, j, 1 + i, 1 + i])

        # Remove the keypoints in consecutive matches that has terminated
        continues_keypoints = [x for x in continues_keypoints if x not in terminated_keypoints]
    terminated_keypoints.extend(continues_keypoints)
    return terminated_keypoints


def plot_timeline(processed_data: np.ndarray):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots()
    for idx, item in enumerate(processed_data):
        # Convert intervals to frames
        start_frame = (item[2] - 1) * FRAMES_PER_INTERVAL
        end_frame = item[3] * FRAMES_PER_INTERVAL
        color = default_colors[idx % len(default_colors)]
        ax.hlines(idx, start_frame, end_frame, colors=color, lw=2)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Keypoint")
    ax.set_title("Timeline")
    # ax.set_yticks(np.arange(0, len(processed_data)))
    max_frame = FRAMES_PER_INTERVAL * MAX_INTERVALS
    ax.set_xlim(0, max_frame)

    # Setting gridlines at the start of each interval
    # ax.set_xticks(np.arange(0, max_frame + 1, FRAMES_PER_INTERVAL))

    # # Shade every frame section
    # for i in range(1, max_frame + 1):
    #     if i % 2 == 0:
    #         ax.axvspan(i - 1, i, facecolor="lightgray", alpha=0.5)

    plt.grid()
    plt.show()
    # Save plot
    fig.savefig("timeline.pdf")

if __name__ == "__main__":
    # Load the descriptors from the file
    print("Loading descriptors...")
    descriptors = load_kpt_des()
    print(len(descriptors), "descriptors loaded.")
    # Process the data to get the matches between each pair of intervals
    print("Processing data...")
    all_matches = process_data(descriptors)
    print(len(all_matches), "pairs of matches found.")
    print("Comparing matches...")
    # Compare the matches between each pair of intervals
    processed_data = compare_matches(all_matches, descriptors)
    print(len(processed_data), "pairs of continuous matches found.")
    print("Plotting timeline...")
    # Plot the timeline
    plot_timeline(processed_data)   