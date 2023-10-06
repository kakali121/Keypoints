"""
Author       : Hanqing Qi
Date         : 2023-08-12 10:39:47
LastEditors  : Hanqing Qi
LastEditTime : 2023-09-20 16:46:51
FilePath     : /WF_RIGHT_2/Robot.py
Description  : This is the class for the robot
"""

### Import Packages ###
from Optitrack_dependency.util import quaternion_to_euler_angle_vectorized1
from Optitrack_dependency.NatNetClient import NatNetClient
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import socket

### Constants ###
gain_v = 35        #30 25 24
gain_ω = -28        #20 40 50
bound_v = 20
bound_ω = 17
#24 -53 25 10
OPT_CLIENT_ADDRESS = "192.168.0.52"

ZERO_COMMAND = "CMD_MOTOR#0#0#0#0\n"
DEMO_DATA = "./Results/record_data.txt"
INIT_OPTITRACK = False


class Robot:
    def __init__(self, IP_adress: str, connect: bool, show_trace: bool, savepath: str) -> None:
        # Connection parameters
        self.IP_adress = IP_adress
        self.connected = False
        self.clientAddress = OPT_CLIENT_ADDRESS
        self.optitrackServerAddress = "192.168.0.4"
        self.robot_id = 121
        self.positions = {}
        self.rotations = {}

        # Optitrack parameters
        if INIT_OPTITRACK:
            if not self.optitrack_init():
                print("### Optitrack is cannot be initialized ###")
            print("### Optitrack is initialized ###")
            self.demo_trace = []
            self.x_data = []
            self.y_data = []
            self.file = open(savepath + "/test_data.txt", "w")  # Open the file for writing
        else:
            show_trace = False
            print("### Optitrack is disabled - Trace plot automatically set of false ###")

        if show_trace:
        # Create figure and axis for plotting
            fig, self.ax = plt.subplots(figsize=(4, 3))
            self._load_demo_trace()
            self.ln, = self.ax.plot([], [], 'r-')  # Start with empty data
            
            # Initialize the animation
            self.animation_running = True
            self.ani = FuncAnimation(fig, self.update, frames=None, interval=5, blit=True)
        else:
            self.animation_running = False

        # Record data
        if connect:
            self.connect()

    def __str__(self) -> str:
        pass

    def update(self, frame):
        if self.animation_running:  # Only proceed with updates if this flag is True
            self._record_location()
            return self.ln,
        else:
            return self.ln,

    def optitrack_init(self) -> None:
        streaming_client = NatNetClient()
        streaming_client.set_client_address(self.clientAddress)
        streaming_client.set_server_address(self.optitrackServerAddress)
        streaming_client.set_use_multicast(True)
        streaming_client.rigid_body_listener = self._receive_rigid_body_frame
        is_running = streaming_client.run()
        return is_running

    def _receive_rigid_body_frame(self, robot_id, position, rotation_quaternion):
        # Position and rotation received
        self.positions[robot_id] = position
        # The rotation is in quaternion. We need to convert it to euler angles
        rotx, roty, rotz = quaternion_to_euler_angle_vectorized1(rotation_quaternion)
        self.rotations[robot_id] = rotz

    def connect(self) -> None:
        """
        @description: Connect to the robot
        @param       {*} self: -
        @return      {*}: None
        """
        self.connected = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.IP_adress, 5000))
        print("### The robot is connected ###")

    def _record_location(self) -> None:
        """
        @description: Record the location of the robot
        @param       {*} self: -
        @return      {*}: None
        """
        if self.robot_id in self.positions:
            x = self.positions[self.robot_id][0]
            y = self.positions[self.robot_id][1]
            # For plotting
            self.x_data.append(x)
            self.y_data.append(y)
            self.ln.set_data(self.x_data, self.y_data)
            rotation = self.rotations[self.robot_id]
            self.file.write(f"{0:.2f}, {x}, {y}, {rotation}\n")
            print("x:", x, "y:", y, " rotation:", rotation)

    def move(self, v: float, ω: float) -> None:
        """
        @description: Move the robot based on the linear and angular velocity
        @param       {*} self: -
        @param       {float} v: linear velocity
        @param       {float} ω: angular velocity
        @return      {*}: None
        """

    def move_legacy(self, v: float, ω: float) -> None:
        if not self.animation_running:
            self._record_location()
        pwm = [0, 0]
        v = 0 if -bound_v < v < bound_v else min(max(v, -bound_v), bound_v)
        ω = 0 if -bound_ω < ω < bound_ω else min(max(ω, -bound_ω), bound_ω)
        ω = -ω if v < 0 else ω
        pwm[0] = int(gain_v * v + gain_ω * ω)  # For left wheel
        pwm[1] = int(gain_v * v - gain_ω * ω)  # For right wheel
        command = "CMD_MOTOR#%d#%d#%d#%d\n" % (pwm[0], pwm[0], pwm[1], pwm[1])
        if self.connected:
            self.socket.send(command.encode())
            print("@sending: ", command)
        else:
            print("@debug: ", command)
        # self.socket.send(ZERO_COMMAND.encode())

    def disconnect(self) -> None:
        """
        description: Disconnect the robot
        param       {*} self: -
        return      {*}: None
        """
        if self.connected:
            self.socket.send(ZERO_COMMAND.encode())
            self.socket.close()
            self.connected = False
        self.animation_running = False  # Stop the animation updates
        print("### The robot is disconnected ###")

    def _load_demo_trace(self) -> None:
        try:
            with open(DEMO_DATA, 'r') as file:
                for line in file:
                    # Split the line into values using comma as a delimiter
                    values = line.split(',')
                    
                    # Extract x and y values
                    x = float(values[1].strip())
                    y = float(values[2].strip())
                    
                    # Append the x, y tuple to the data_points list
                    self.demo_trace.append((x, y))
                # Trim demo trace
                self.demo_trace = self.demo_trace[:7000]
            self._draw_demo_trace()
        except Exception as e:
            print("Get exception: ", e)

    def _draw_demo_trace(self):
        x_vals, y_vals = zip(*self.demo_trace)
        # Draw the demo trace
        self.ax.set_xlim(-2, 5)
        self.ax.set_ylim(-3, 2)
        self.ax.plot(x_vals, y_vals, '-', color='blue', linewidth=2)