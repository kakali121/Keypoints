"""
Author       : Hanqing Qi
Date         : 2023-08-15 18:02:17
LastEditors  : Hanqing Qi
LastEditTime : 2023-08-15 18:02:17
FilePath     : /WallFollowing_Front/Timeline/revert_video.py
Description  : Revert a video
"""
import cv2

def reverse_video(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Check if video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read all frames and store them in a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Write frames in reverse order
    for frame in reversed(frames):
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Reversed video saved to {output_path}")

# Example usage
input_path = 'frontdemo_original.mp4'
output_path = 'frontdemo.mp4'
reverse_video(input_path, output_path)
