import cv2
import time
from Optitrack_dependency.NatNetClient import NatNetClient
from Optitrack_dependency.util import quaternion_to_euler_angle_vectorized1

positions = {}
rotations = {}

NAME = 'corner.mp4'
INIT_OPTITRACK = False

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame(robot_id, position, rotation_quaternion):
    # Position and rotation received
    positions[robot_id] = position
    # The rotation is in quaternion. We need to convert it to euler angles
    rotx, roty, rotz = quaternion_to_euler_angle_vectorized1(rotation_quaternion)
    rotations[robot_id] = rotz

if __name__ == "__main__":
    print("##### WARNING: This program will overwrite the previous record data #####")
    input("Press Enter to start recording...")
    clientAddress  = "192.168.0.66"
    optitrackServerAddress = "192.168.0.4"
    robot_id = 121

    is_running = False
    if INIT_OPTITRACK:
        # This will create a new NatNet client
        streaming_client = NatNetClient()
        streaming_client.set_client_address(clientAddress)
        streaming_client.set_server_address(optitrackServerAddress)
        streaming_client.set_use_multicast(True)
        streaming_client.rigid_body_listener = receive_rigid_body_frame
        is_running = streaming_client.run()

    cap = cv2.VideoCapture("http://192.168.0.204:1234/stream.mjpg")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    out = cv2.VideoWriter(NAME, fourcc, fps, (400, 300))

    with open('record_data.txt', 'w') as file:
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if not is_running and INIT_OPTITRACK:
                print("### Optitrack is not running ###")
                break

            if INIT_OPTITRACK and robot_id in positions:  # Only retrieve position if streaming is enabled
                x = positions[robot_id][0]
                y = positions[robot_id][1]
                rotation = rotations[robot_id]
                elapsed_time = time.time() - start_time
                file.write(f"{elapsed_time:.2f}, {x}, {y}, {rotation}\n")
                print('x:', x, 'y:', y, ' rotation:', rotation)
                print(f"Elapsed Time: {elapsed_time:.2f} seconds")

            out.write(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord('q'):
                if INIT_OPTITRACK:
                    streaming_client.shutdown()
                cv2.destroyAllWindows()
                cap.release()
                break
