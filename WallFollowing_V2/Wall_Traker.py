'''
Author       : Karen Li
Date         : 2023-08-12 14:27:18
LastEditors  : Karen Li
LastEditTime : 2023-08-12 19:09:21
FilePath     : /WallFollowing_V2/Wall_Traker.py
Description  : Wall traker of the robot
'''

### Import Packages ###
from typing import List, Tuple
from State import State
import numpy as np
import math
import cv2

### Constants ###
DEMO_VIDEO = "side_demo.mp4"                # The path to the demo video

class Wall_Traker:
    def __init__(self, initial_frame: np.array, total_interval: int, interval_length: int, skip_interval: int) -> None:
        self.total_interval = total_interval # Total number of intervals in the demo video
        self.interval_length = interval_length # Number of frames in a timeline interval
        self.skip_interval = skip_interval # Interval between donkey and carrot
        
        self.accumulated_y_ratio = 0.0 # Accumulated y ratio

        self.total_states = self._load_all_states() # Load all frames from the demo video into a list of states
        self.robot_state = State(initial_frame) # Create a state object for the robot
        self._find_donkey_carrot_state() # Find the donkey and carrot state
        self.donkey_index = -1 # The index of the donkey state
        self.carrot_index = -1 # The index of the carrot state
        

    def __str__(self) -> str:
        pass

    def _load_all_states(self) -> List[State]:
        '''
        description: Load all frames from the demo video into a list of states
        param       {*} self: -
        return      {*}: None
        '''
        # Create a VideoCapture object to read the video file
        video = cv2.VideoCapture(DEMO_VIDEO)
        for index in range(self.total_interval):
            # Read a frame from the video
            video.set(cv2.CAP_PROP_POS_FRAMES, (index+1)*self.interval_length)
            ret, frame = video.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Create a state object
            state = State(frame, load=True, interval=index+1)
            self.total_states.append(state)
        video.release()


    def _find_donkey_carrot_state(self):
        '''
        description: Find the donkey and carrot state
        param       {*} self: -
        return      {*}: None
        '''
        # Find the state with the least distance to the robot state
        min_distance = math.inf
        self.carrot_index = -1
        self.donkey_index = -1
        for index, state in enumerate(self.total_states):
            distance = self.robot_state.distance(state)
            if distance < min_distance:
                min_distance = distance
                self.donkey_index = index
                self.carrot_index = index + self.skip_interval
        self.donkey_state = self.total_states[self.donkey_index]
        self.carrot_state = self.total_states[self.carrot_index]

    def _calculate_moving_average_y(self, new_y_ratio: float) -> float:
        '''
        description: Calculate the moving average of the y ratio
        param       {*} self: -
        param       {float} new_y_ratio: The current y ratio
        return      {float} The moving average of the y ratio
        '''
        if self.accumulated_y_ratio == 0: # If the accumulated y ratio is 0, set it to the current y ratio
            self.accumulated_y_ratio = new_y_ratio
        else: 
            # Calculate the difference between the current y ratio and the accumulated y ratio
            y_ratio_diff = abs((new_y_ratio - self.accumulated_y_ratio) / self.accumulated_y_ratio) 
            if y_ratio_diff > 10: # If the difference is too big, discard the current y ratio
                print("Warning: Broken Match!")
                print("Discard y ratio: " + str(new_y_ratio))
                return self.accumulated_y_ratio
            # The dynamic gain is the exponential of the difference
            dynamic_gain = 1/math.exp(y_ratio_diff) 
            # Calculate the new accumulated y ratio
            self.accumulated_y_ratio = self.accumulated_y_ratio * (1-dynamic_gain) + new_y_ratio * dynamic_gain
        return self.accumulated_y_ratio
    
    def chase_carrot(self)-> Tuple[int, float]:
        '''
        description: Let robot chase the carrot
        param       {*} self: -
        return      {Tuple[float, float]}: The linear velocity and angular velocity of the robot
        '''
        query_coordinte, train_coordinate = self.robot_state.get_match_coordinate(self.carrot_state)
        # If no match is found, return 0 velocity
        if query_coordinte == [] or train_coordinate == []:
            print("No match found!")
            return 0, 1
        # Calculate the average x and y difference
        robot_center_x, robot_center_y, robot_x_radiu, robot_y_radiu = self.robot_state.compute_confidence_ellipse()
        carrot_center_x, carrot_center_y, carrot_x_radiu, carrot_y_radiu = self.carrot_state.compute_confidence_ellipse()
        x_diff = carrot_center_x - robot_center_x
        y_ratio = robot_y_radiu / carrot_y_radiu
        # Calculate the moving average of the y ratio
        processed_y_ratio = self._calculate_moving_average_y(y_ratio)
        return x_diff, processed_y_ratio
    
    def update_carrot(self) -> None:
        '''
        description: Update the carrot state
        param       {*} self: -
        return      {*}: None
        '''
        self.carrot_state = self.total_states[self.carrot_index + self.skip_interval]

    def update_robot(self, new_frame: np.array) -> None:
        '''
        description: Update the robot state
        param       {*} self: -
        return      {*}: None
        '''
        self.robot_state = State(new_frame)
        