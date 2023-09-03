import cv2 
import numpy as np

FILENAME = 'circle.npy'
NAME = 'circular.mp4'

if __name__ == "__main__":
    # Load the .npy file
    frames = np.load('video.npy')
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # specify the video codec
    fps = 20
    out = cv2.VideoWriter(NAME, fourcc, fps, (400, 300))

    # Write frames to the video writer
    for frame in frames:
        out.write(frame)

    # Release the video writer
    out.release()




