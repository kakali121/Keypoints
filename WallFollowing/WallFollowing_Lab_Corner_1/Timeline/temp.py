"""
Author       : Hanqing Qi
Date         : 2023-09-03 11:25:37
LastEditors  : Hanqing Qi
LastEditTime : 2023-09-04 10:31:31
FilePath     : /undefined/Users/hanqingqi/Library/CloudStorage/Dropbox/Keypoints/WallFollowing/WallFollowing_Lab_Corner/Timeline/temp.py
Description  : Temporary file for testing
"""

import matplotlib.pyplot as plt
import numpy as np


def process_matches(testdata):
    debug1 = []
    debug2 = []

    for i, matches in enumerate(testdata):
        match_pair_1, match_pair_2 = zip(*matches)

        last_matches = [x[1] for x in debug1]

        for j in range(10):
            if j in match_pair_1:
                if j in last_matches:
                    index = last_matches.index(j)
                    debug1[index][1] = match_pair_2[match_pair_1.index(j)]
                    debug1[index][3] = i + 2
                else:
                    debug1.append(
                        [j, match_pair_2[match_pair_1.index(j)], i + 1, i + 2]
                    )
            else:
                if j in last_matches:
                    debug2.append(debug1[last_matches.index(j)])
                else:
                    debug2.append([j, j, 1 + i, 1 + i])
        debug1 = [x for x in debug1 if x not in debug2]
    debug2.extend(debug1)
    return debug2

def plot_timeline(data, max_intervals=10, frames_per_interval=12):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots()
    for idx, item in enumerate(data):
        # Convert intervals to frames
        start_frame = (item[2] - 1) * frames_per_interval
        end_frame = item[3] * frames_per_interval
        
        color = default_colors[idx % len(default_colors)]
        ax.hlines(idx, start_frame, end_frame, colors=color, lw=2)
    
    ax.set_xlabel("Frame")
    ax.set_ylabel("Keypoint")
    ax.set_title("Timeline")
    ax.set_yticks(np.arange(0, len(data)))

    max_frame = frames_per_interval * max_intervals
    ax.set_xlim(0, max_frame)
    
    # Setting gridlines at the start of each interval
    ax.set_xticks(np.arange(0, max_frame + 1, frames_per_interval))
    
    # Shade every frame section
    for i in range(1, max_frame + 1):
        if i % 2 == 0:
            ax.axvspan(i - 1, i, facecolor='lightgray', alpha=0.5)

    plt.grid()
    plt.show()


def plot_keypoints_timeline(data, max_intervals=10, frames_per_interval=12):
    processed_data = process_matches(data)
    plot_timeline(processed_data, max_intervals, frames_per_interval)


if __name__ == "__main__":
    # Example usage:
    data = [
        [(1, 3), (2, 4), (3, 5), (5, 9)],
        [(3, 4), (4, 5), (8, 9)],
        [(1, 2), (2, 3), (4, 5), (5, 6)],
        [(2, 4), (4, 5), (5, 6), (6, 7)],
        [(1, 1), (2, 3), (3, 5), (4, 6), (6, 9)],
    ]
    max_intervals = 10
    frames_per_interval = 12
    plot_keypoints_timeline(data, max_intervals, frames_per_interval)