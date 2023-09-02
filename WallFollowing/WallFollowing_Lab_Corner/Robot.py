'''
Author       : Hanqing Qi
Date         : 2023-08-12 10:39:47
LastEditors  : Karen Li
LastEditTime : 2023-08-26 15:38:15
FilePath     : /WallFollowing_Corner/Robot.py
Description  : This is the class for the robot
'''

### Import Packages ###
import numpy as np
import socket

### Constants ###
MIN_V = 700 # Minimum velocity of the robot to overcome friction
MAX_V = 900 # Maximum velocity of the robot 
BANGBANG_V = [800, 700, 800]
ZERO_COMMAND = 'CMD_MOTOR#0#0#0#0\n' # Command to stop the robot

class Robot:
    def __init__(self, IP_adress: str, connect: bool) -> None:
        self.IP_adress = IP_adress
        self.connected = False
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
        self.connected = True
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
        if v == 0:
            command = ZERO_COMMAND
        else:
            # Implement bang-bang control
            if v > 30:
                if ω > 40:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(-BANGBANG_V[0], -BANGBANG_V[0], BANGBANG_V[0], BANGBANG_V[0])
                elif ω < -40:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(BANGBANG_V[0], BANGBANG_V[0], -BANGBANG_V[0], -BANGBANG_V[0])
                else:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(BANGBANG_V[1], BANGBANG_V[1], BANGBANG_V[1], BANGBANG_V[1])
            elif v < -30:
                if ω > 40:
                    command = 'CMD_MOTOR#%d#%d#%d#%d\n'%(BANGBANG_V[0], BANGBANG_V[0], -BANGBANG_V[0], -BANGBANG_V[0])
                elif ω < -40:
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

    
    