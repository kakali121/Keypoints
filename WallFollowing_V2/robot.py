'''
Author       : Hanqing Qi
Date         : 2023-08-12 10:39:47
LastEditors  : Hanqing Qi
LastEditTime : 2023-08-12 11:12:48
FilePath     : /WallFollowing_V2/robot.py
Description  : This is the class for the robot
'''

### Import Packages ###
import numpy as np
import socket

### Constants ###
MIN_V = 520 # Minimum velocity of the robot to overcome friction
MAX_V = 800 # Maximum velocity of the robot 

class Robot:
    def __init__(self, IP_adress: str, connect: bool) -> None:
        self.IP_adress = IP_adress
        if connect:
            self.connect()
        
    def __str__(self) -> str:
        pass

    def connect(self) -> None:
        '''
        description: Connect to the robot
        param       {*} self: -
        return      {*}: None
        '''
        self.connect = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.IP_adress, 5000))
        print('### The robot is connected ###')

    def move(self, v: float, ω: float) -> None:
        '''
        description: Move the robot based on the linear and angular velocity
        param       {*} self: -
        param       {float} v: linear velocity
        param       {float} ω: angular velocity
        return      {*}: None
        '''
        # Calculate the left and right wheel velocity
        control_param = np.array([v - ω, v + ω])
        # Limit the velocity to the range of [MIN_V, MAX_V]
        control_param = [min(MAX_V, max(MIN_V, p)) if p >= 0 else max(-MAX_V, min(-MIN_V, p)) for p in control_param]
        # Construct the command
        command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(control_param[0], control_param[0], control_param[1], control_param[1])
        if self.connect:
            self.socket.send(command.encode()) # Send the command to the robot
        else:
            print('@debug: ', command) # Print the command if the robot is not connected

    def disconnect(self) -> None:
        '''
        description: Disconnect the robot
        param       {*} self: -
        return      {*}: None
        '''
        if self.connect:
            command = 'CMD_MOTOR#0#0#0#0\n'
            self.socket.send(command.encode())
            self.socket.close()
            self.connect = False
        print('### The robot is disconnected ###')

    
    