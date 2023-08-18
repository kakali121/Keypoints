"""
Author       : Hanqing Qi
Date         : 2023-08-12 10:39:47
LastEditors  : Hanqing Qi
LastEditTime : 2023-08-15 14:28:24
FilePath     : /WallFollowing_Front/Robot.py
Description  : This is the class for the robot
"""

### Import Packages ###
import numpy as np
import socket

### Constants ###
# Robot Velocity Limits
MIN_V = 700  
MAX_V = 900  

# Bang-bang control values
BANGBANG_V = [700, 650, 800]

# Robot Commands
ZERO_COMMAND = "CMD_MOTOR#0#0#0#0\n"


class Robot:
    def __init__(self, IP_address: str, connect: bool) -> None:
        """
        description: The constructor of the robot class
        param       {*} self: -
        param       {str} IP_address: IP address of the robot
        param       {bool} connect: Whether to connect to the robot
        return      {*}: None
        """
        self.IP_address = IP_address
        self.connected = False
        if connect:
            self.connect()

    def __str__(self) -> str:
        """
        description: The string representation of the robot
        param       {*} self: -
        return      {str}: The string representation of the robot
        """
        return f"Robot IP: {self.IP_address}, Connected: {self.connected}"

    def connect(self) -> None:
        """
        description: Connect to the robot
        param       {*} self: -
        return      {*}: None
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.IP_address, 5000))
            self.connected = True
            print("#-------- The robot is connected --------#")
        except Exception as e:
            print(f"Error connecting to robot: {e}")

    def construct_command(self, velocities: list) -> str:
        """
        description: Construct the command to send to the robot
        param       {*} self: -
        param       {list} velocities: The velocities of the four wheels
        return      {str}: The command to send to the robot
        """
        return f"CMD_MOTOR#{velocities[0]}#{velocities[1]}#{velocities[2]}#{velocities[3]}\n"

    def move(self, v: float, ω: float) -> None:
        """
        description: Move the robot with the given linear and angular velocity
        param       {*} self: -
        param       {float} v: Linear velocity
        param       {float} ω: Angular velocity
        return      {*}: None
        """
        if abs(v) < 20:
            command = ZERO_COMMAND
        else:
            idx = 0 if abs(ω) > 60 else 1
            base = -1 if v < 0 else 1
            rotation = 1 if ω > 0 else -1
            command_values = [BANGBANG_V[idx] * base * (1 - rotation), 
                              BANGBANG_V[idx] * base * (1 + rotation)] * 2
            command = self.construct_command(command_values)

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
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
            self.connected = False
            print("#-------- The robot is disconnected --------#")
        else:
            print("Robot is already disconnected")