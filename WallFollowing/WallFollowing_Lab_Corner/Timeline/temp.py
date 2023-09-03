'''
Author       : Hanqing Qi
Date         : 2023-09-03 11:25:37
LastEditors  : Hanqing Qi
LastEditTime : 2023-09-03 11:39:54
FilePath     : /Timeline/temp.py
Description  : 
'''

import matplotlib.pyplot as plt
import numpy as np

debug1 = []
debug2 = []
testdata = [
    [(1, 3), (2, 4), (3, 5), (5, 9)],
    [(3, 4), (4, 5), (8, 9)],
    [(1, 2), (2, 3), (4, 5), (5, 6)],
    [(2, 4), (4, 5), (5, 6), (6, 7)],
    [(1, 1), (2, 3), (3, 5), (4, 6)]
]
for i in range(0, 5):
    matches = testdata[i]
    print("matches: ", matches)
    match_pair_1 = [match[0] for match in matches]
    match_pair_2 = [match[1] for match in matches]
    print("match_pair_1: ", match_pair_1)
    print("match_pair_2: ", match_pair_2)
    last_matches = [x[1] for x in debug1]
    print("last_matches: ", last_matches)
    for j in range(10):
        if j in match_pair_1:
            # Check if the matched keypoint is also matched by another keypoint
            if j in last_matches:
                # If so, update the last match of the keypoint
                debug1[last_matches.index(j)][1] = match_pair_2[match_pair_1.index(j)]
                debug1[last_matches.index(j)][3] = i+2
            else:
                debug1.append([j,match_pair_2[match_pair_1.index(j)],i+1,i+2])
        else:
            if j in last_matches:
                debug2.append(debug1[last_matches.index(j)])
            else:
                # print("j: ", j)
                debug2.append([j,j,1+i,1+i])
    # Remove elements in debug2 from debug1
    # debug1 = [x for x in debug1 if x not in debug2]
    # Add elements in debug1 to debug2
    # print("debug1: ", debug1)
    # print("debug2: ", debug2)

debug2.extend(debug1)
print("Final list of elements: ", debug2)

# Plot the lines in debug2
# The line goes from its starting interval to its ending interval and is horizontal, but each line should be seperated by 1 unit on y-axis, no lines should overlap
# The color of the line is randomly chosen from the 10 colors in matplotlib
# Plot using matplotlib
# The interval have a length of 1, thus if the interval is (1, 2), the line will be plotted from the start of the first interval to the end of the second interval which is x=1 to x=3
# If a line has interval (n,n), it should stil last the whole interval, thus the line will be plotted from x=n to x=n+1

# Extract default colors from matplotlib
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Generate the plot
fig, ax = plt.subplots()
for idx, item in enumerate(debug2):
    start_interval = item[2]
    end_interval = item[3] + 1
    color = default_colors[idx % len(default_colors)]
    ax.hlines(idx, start_interval, end_interval, colors=color, lw=2)

ax.set_xlabel('Interval')
ax.set_ylabel('Keypoint')
ax.set_title('Timeline')
ax.set_yticks(np.arange(0, len(debug2)))
ax.set_xticks(np.arange(0, 10, 1))
ax.set_ylim(-1, len(debug2))
ax.set_xlim(0, 10)
plt.show()