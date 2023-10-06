import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def process_data(data):
    data = data[data[:, 1] != 0]  # Remove all the data where y = 0
    data = data[200:5000]
    return data

# Step 1: Load the NumPy file
def load_numpy_file(file_path):
    try:
        data = np.load(file_path)
        data = data[data[:, 1] != 0]  # Remove all the data where y = 0
        data = data[200:5000]
        return data 
    except Exception as e:
        print("Error loading NumPy file:", e)
        return None
    
# Step 2: Create a Matplotlib line plot
def create_line_plot(data):
    plt.rc("font", family = "serif")
    plt.rc("text", usetex = True)
    plt.rc("xtick", labelsize = 24)
    plt.rc("ytick", labelsize = 24)
    plt.rc("figure", figsize = (12, 4))
    plt.rc("legend", fontsize = 18)
    plt.rc("axes", titlesize = 30)
    plt.rc("axes", labelsize = 32)
    if data is not None:
        plt.plot(np.linalg.norm(data, axis=1), linestyle='-')

        # Plot the mean of the data
        plt.ylabel('Ellipse Mean Error (px)')  # Customize the Y-axis label
        plt.xlim(0, 4800)            # Customize the X-axis range
        plt.ylim(0, 100)            # Customize the Y-axis range
        plt.savefig('SB_Error_Plot.pdf', bbox_inches='tight')
        plt.show()
        # plt.show()

def init(line):
    line.set_data([], [])
    return line,

def update(frame, line, data):
    line.set_data(np.arange(frame), np.linalg.norm(data[:frame], axis=1))
    progress_bar.update(1)
    return line,

def create_animation(data, interval):
    fig, ax = plt.subplots(figsize=(12, 4))

    # Apply rc configurations
    plt.rc("font", family="serif")
    plt.rc("text", usetex=True)
    plt.rc("xtick", labelsize=24)
    plt.rc("ytick", labelsize=24)
    plt.rc("figure", figsize=(12, 4))
    plt.rc("legend", fontsize=18)
    plt.rc("axes", titlesize=30)
    plt.rc("axes", labelsize=30)

    ax.set_xlim(0, len(data))
    ax.set_ylim(0, 100)  # Assuming this is the desired y-axis range
    ax.set_ylabel('Ellipse Mean Error (px)')

    line, = ax.plot([], [], linestyle='-')
    
    global progress_bar
    progress_bar = tqdm.tqdm(total=len(data), desc="Animating", position=0, leave=True)
    
    ani = FuncAnimation(fig, update, frames=len(data), init_func=lambda: init(line), blit=True, interval=interval, fargs=(line, data,), repeat=False)
    # Optionally save the animation to a file before showing
    ani.save("SB_Errors_animation.mp4", writer="ffmpeg", extra_args=['-vcodec', 'libx264']) 
    plt.show()


if __name__ == "__main__":
    # Specify the path to your NumPy file
    numpy_file_path = "SB_Error.npy"
    
    # Step 1: Load the NumPy file
    loaded_data = load_numpy_file(numpy_file_path)
    print(np.shape(loaded_data))
    # Step 2: Create a Matplotlib line plot
    if loaded_data is not None:
        create_line_plot(loaded_data)
        total_time = 245.79 * 1000  # Total time for the animation in milliseconds
        interval = total_time / len(loaded_data)  # Interval between frames in milliseconds
        create_animation(loaded_data, interval)