import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- DTW Function Definitions ---
def euclidean_distance(p, q):
    return np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

def dtw(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dtw_matrix = np.full((m+1, n+1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in tqdm(range(1, m+1), desc="Computing DTW", ncols=100):
        for j in range(1, n+1):
            cost = euclidean_distance(seq1[i-1], seq2[j-1])
            dtw_matrix[i, j] = cost + min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
    
    # Extracting the warping path for local errors
    path = []
    i, j = m, n
    while i > 0 or j > 0:
        path.append((i, j))
        min_index = np.argmin([dtw_matrix[i-1, j-1], dtw_matrix[i, j-1], dtw_matrix[i-1, j]])
        if min_index == 0:
            i, j = i-1, j-1
        elif min_index == 1:
            i, j = i, j-1
        else:
            i, j = i-1, j
    path.reverse()
    
    # Computing local errors using the warping path
    local_errors = [euclidean_distance(seq1[i-1], seq2[j-1]) for i, j in path]
    
    return dtw_matrix[m, n], local_errors

# --- Data Extraction ---
def extract_data_from_file(filename):
    data_points = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.split(',')
            x = float(values[1].strip())
            y = float(values[2].strip())
            data_points.append((x, y))
    return data_points

# Global configurations
FOLDERS = ['1', '2', '3', '4', '5', '6']
RECORD_DATA_CUTOFF = 4200
X_LIM = (0, 4)
Y_LIM = (-2.5, 2)
ERRORS = []
LOCAL_ERRORS = []

# Read and plot record_data.txt as reference
record_data_points = extract_data_from_file('record_data.txt')
record_data_points = record_data_points[:RECORD_DATA_CUTOFF]

fig, ax = plt.subplots()

# Plot record_data.txt in red
ax.plot(*zip(*record_data_points), 'r-', label="Record Data")

# Loop through folders and plot test_data.txt in blue, compute DTW and store error
for folder in FOLDERS:
    folder_path = os.path.join(folder, 'test_data.txt')
    test_data_points = extract_data_from_file(folder_path)
    
    # Compute DTW error between record data and current test data
    error, local_err = dtw(record_data_points, test_data_points)
    ERRORS.append(error)
    LOCAL_ERRORS.append(local_err)
    
    # Plot the test data
    ax.plot(*zip(*test_data_points), 'b-', label=f"Test {folder}")


# Print the computed errors
for i, error in enumerate(ERRORS):
    print(f"Overall Error for Test {FOLDERS[i]}: {error:.2f}")
    # print(f"Local Errors for Test {FOLDERS[i]}: {LOCAL_ERRORS[i]}")

# Display the plot
ax.legend(loc="best")
plt.savefig('./Errors/overall_dtw_errors.png', dpi=300)
plt.show()

# Plotting the local DTW errors for each test dataset
for idx, local_err in enumerate(LOCAL_ERRORS):
    plt.figure(figsize=(10,6))
    plt.plot(local_err, color='skyblue')
    plt.xlabel('Warping Path Index')
    plt.ylabel('Local Error')
    plt.title(f'Local DTW Errors for Test {FOLDERS[idx]}')
    # Save the plot
    plt.savefig(f'./Errors/local_dtw_errors_test_{FOLDERS[idx]}.png', dpi=300)
    plt.show()