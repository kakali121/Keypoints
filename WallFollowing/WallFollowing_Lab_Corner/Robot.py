'''
Author       : Hanqing Qi
Date         : 2023-08-12 10:39:47
LastEditors  : Hanqing Qi
LastEditTime : 2023-09-03 19:58:35
FilePath     : /WallFollowing_Lab_Corner/Robot.py
Description  : This is the class for the robot
'''

### Import Packages ###
from Optitrack_dependency.NatNetClient import NatNetClient
from Optitrack_dependency.util import quaternion_to_euler_angle_vectorized1
import numpy as np
import socket

### Constants ###
MIN_V = 700 # Minimum velocity of the robot to overcome friction
MAX_V = 900 # Maximum velocity of the robot 
BANGBANG_V = [1100, 800, 800]
ZERO_COMMAND = 'CMD_MOTOR#0#0#0#0\n' # Command to stop the robot

class Robot:
    def __init__(self, IP_adress: str, connect: bool) -> None:
        self.IP_adress = IP_adress
        self.connected = False
        self.clientAddress = "192.168.0.46"
        self.optitrackServerAddress = "192.168.0.4"
        self.robot_id = 121
        if not self.optitrack_init():
            print('### Optitrack is cannot be initialized ###')
        self.file = open('test_data.txt', 'w')  # Open the file in write mode
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
        '''
        description: Connect to the robot
        param       {*} self: -
        return      {*}: None
        '''
        self.connected = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.IP_adress, 5000))
        print('### The robot is connected ###')

    def _record_location(self) -> None:
        if self.robot_id in self.positions:
            x = self.positions[self.robot_id][0]
            y = self.positions[self.robot_id][1]
            rotation = self.rotations[self.robot_id]
            self.file.write(f"{0:.2f}, {x}, {y}, {rotation}\n")
            print('x:',x, 'y:',y, ' rotation:', rotation)

    def move(self, v: float, ω: float) -> None:
        '''
        description: Move the robot based on the linear and angular velocity
        param       {*} self: -
        param       {float} v: linear velocity
        param       {float} ω: angular velocity
        return      {*}: None
        '''
        self._record_location()
        if v == 0 and ω == 0:
            command = ZERO_COMMAND
        else:
            if v > 30:
                Base = 600 + int(v) + abs(int(ω))
                if ω > 20:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(-Base, -Base, Base, Base)
                elif ω < -20:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(Base, Base, -Base, -Base)
                else:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(Base, Base, Base, Base)
            elif v < -30:
                Base = -600 + int(v) - abs(int(ω))
                if ω > 20:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(Base, Base, -Base, -Base)
                elif ω < -20:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(-Base, -Base, Base, Base)
                else:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(-Base, -Base, -Base, -Base)
            else:
                command = ZERO_COMMAND
        if self.connected:
            self.socket.send(command.encode()) # Send the command to the robot
            print('@sending: ', command) # Print the command if the robot is connected
        else:
            print('@debug: ', command) # Print the command if the robot is not connected

    def move_legacy(self, v: float, ω: float) -> None:
        self._record_location()
        if v == 0:
            command = ZERO_COMMAND
        else:
            # Implement bang-bang control
            if v > 20:
                if ω > 30:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(-BANGBANG_V[0], -BANGBANG_V[0], BANGBANG_V[0], BANGBANG_V[0])
                elif ω < -30:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(BANGBANG_V[0], BANGBANG_V[0], -BANGBANG_V[0], -BANGBANG_V[0])
                else:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(BANGBANG_V[1], BANGBANG_V[1], BANGBANG_V[1], BANGBANG_V[1])
            elif v < -20:
                if ω > 30:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(BANGBANG_V[0], BANGBANG_V[0], -BANGBANG_V[0], -BANGBANG_V[0])
                elif ω < -30:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(-BANGBANG_V[0], -BANGBANG_V[0], BANGBANG_V[0], BANGBANG_V[0])
                else:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(-BANGBANG_V[1], -BANGBANG_V[1], -BANGBANG_V[1], -BANGBANG_V[1])
            else:
                command = ZERO_COMMAND
        if self.connected:
            self.socket.send(command.encode())
            print('@sending: ', command)
        else:
            print('@debug: ', command)

    def disconnect(self) -> None:
        '''
        description: Disconnect the robot
        param       {*} self: -
        return      {*}: None
        '''
        if self.connected:
            self.socket.send(ZERO_COMMAND.encode())
            self.socket.close()
            self.connected = False
        print('### The robot is disconnected ###')

    
    