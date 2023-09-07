'''
Author       : Karen Li
Date         : 2023-08-10 20:45:31
LastEditors  : Karen Li
LastEditTime : 2023-08-26 16:43:35
FilePath     : /WallFollowing_Corner/Keyboard.py
Description  : 
'''

import pygame
import socket
import time
import sys

IP_ADDRESS = '192.168.0.204'

# COMMAND DICTIONARY
COMMANDS = {
    'FORWARD': 'CMD_MOTOR#800#800#800#800\n',
    'BACKWARD': 'CMD_MOTOR#-800#-800#-800#-800\n',
    'LEFT': 'CMD_MOTOR#-1500#-1500#1300#1300\n',
    'RIGHT': 'CMD_MOTOR#1300#1300#-1500#-1500\n',
    'STOP': 'CMD_MOTOR#00#00#00#00\n'
}

# Connect to the robot
def connect_to_robot():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP_ADDRESS, 5000))
    print('Connected')
    return s

def send_command(sock, action):
    sock.send(COMMANDS[action].encode('utf-8'))

def main():
    pygame.init()
    display = pygame.display.set_mode((300, 300))
    
    s = connect_to_robot()

    start_time = time.time()

    try:
        while time.time() - start_time < 600:  # 600 seconds or 10 minutes
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        print("Moving Forward...")
                        send_command(s, 'FORWARD')
                    elif event.key == pygame.K_s:
                        print("Moving Backward...")
                        send_command(s, 'BACKWARD')
                    elif event.key == pygame.K_a:
                        print("Turning Left...")
                        send_command(s, 'LEFT')
                    elif event.key == pygame.K_d:
                        print("Turning Right...")
                        send_command(s, 'RIGHT')
                    elif event.key == pygame.K_t:
                        print("Terminating...")
                        send_command(s, 'STOP')
                    elif event.key == pygame.K_ESCAPE:  # Pressing ESC will terminate the program
                        pygame.quit()
                        sys.exit()
                elif event.type == pygame.KEYUP:  # This event triggers when a key is released
                    send_command(s, 'STOP')
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
    except KeyboardInterrupt:
        send_command(s, 'STOP')
    
    send_command(s, 'STOP')
    s.shutdown(2)
    s.close()

if __name__ == '__main__':
    main()