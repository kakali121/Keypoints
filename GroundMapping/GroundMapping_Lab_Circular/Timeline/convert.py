import cv2 

FILENAME = 'raw.mp4'
NAME = 'circular.mp4'

if __name__ == "__main__":
    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    cap = cv2.VideoCapture(FILENAME)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    out = cv2.VideoWriter(NAME, fourcc, fps, (400, 300))
    while cap.isOpened():
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(f"Video frame dimensions: {width}x{height}")
        # Read a frame from the video
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if not ret: break
        out.write(frame)
        cv2.imshow("frame", frame)
        # Wait for Esc key to stop
        if cv2.waitKey(1) == ord('q'):
            # De-allocate any associated memory usage
            cv2.destroyAllWindows()
            cap.release()
            break
