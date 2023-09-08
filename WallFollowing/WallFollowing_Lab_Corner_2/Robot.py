"""
Author       : Hanqing Qi
Date         : 2023-08-12 10:39:47
LastEditors  : Hanqing Qi
LastEditTime : 2023-09-07 19:46:50
FilePath     : /WallFollowing_Lab_Corner_2/Robot.py
Description  : This is the class for the robot
"""

### Import Packages ###
from Optitrack_dependency.util import quaternion_to_euler_angle_vectorized1
from Optitrack_dependency.NatNetClient import NatNetClient
import socket

### Constants ###
gain_v = 24         #30 25 24
gain_ω = -30        #20 40 50
bound_v = 30
bound_ω = 15
ZERO_COMMAND = "CMD_MOTOR#0#0#0#0\n"


class Robot:
    def __init__(self, IP_adress: str, connect: bool) -> None:
        self.IP_adress = IP_adress
        self.connected = False
        self.clientAddress = "192.168.0.66"
        self.optitrackServerAddress = "192.168.0.4"
        self.robot_id = 121
        if not self.optitrack_init():
            print("### Optitrack is cannot be initialized ###")
        self.file = open("./Results/test_data.txt", "w")  # Open the file in write mode
        self.positions = {}
        self.rotations = {}
        if connect:
            self.connect()

    def __str__(self) -> str:
        pass

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
        description: Connect to the robot
        param       {*} self: -
        return      {*}: None
        """
        self.connected = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.IP_adress, 5000))
        print("### The robot is connected ###")

    def _record_location(self) -> None:
        if self.robot_id in self.positions:
            x = self.positions[self.robot_id][0]
            y = self.positions[self.robot_id][1]
            rotation = self.rotations[self.robot_id]
            self.file.write(f"{0:.2f}, {x}, {y}, {rotation}\n")
            print("x:", x, "y:", y, " rotation:", rotation)

    def move(self, v: float, ω: float) -> None:
        """
        description: Move the robot based on the linear and angular velocity
        param       {*} self: -
        param       {float} v: linear velocity
        param       {float} ω: angular velocity
        return      {*}: None
        """

    def move_legacy(self, v: float, ω: float) -> None:
        self._record_location()
        pwm = [0, 0]
        v = 0 if -bound_v < v < bound_v else min(max(v, -bound_v), bound_v)
        ω = 0 if -bound_ω < ω < bound_ω else min(max(ω, -bound_ω), bound_ω)
        pwm[0] = int(gain_v * v + gain_ω * ω)  # For left wheel
        pwm[1] = int(gain_v * v - gain_ω * ω)  # For right wheel
        command = "CMD_MOTOR#%d#%d#%d#%d\n" % (pwm[0], pwm[0], pwm[1], pwm[1])
        if self.connected:
            self.socket.send(command.encode())
            print("@sending: ", command)
        else:
            print("@debug: ", command)

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
        print("### The robot is disconnected ###")
