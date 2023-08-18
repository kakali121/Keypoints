import cv2 

NAME = 'frontdemo.mp4'

if __name__ == "__main__":
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture("http://192.168.0.204:1234/stream.mjpg")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    out = cv2.VideoWriter(NAME, fourcc, fps, (400, 300))
    try:
        while cap.isOpened():
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
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        cv2.destroyAllWindows()
        cap.release()
