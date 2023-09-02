'''
Author       : Karen Li
Date         : 2023-08-12 14:27:18
LastEditors  : Karen Li
LastEditTime : 2023-09-02 16:17:24
FilePath     : /WallFollowing_Lab_Corner/WallTraker.py
Description  : Wall traker of the robot
'''

### Import Packages ###
from typing import List, Tuple
from State import State
import numpy as np
import math
import cv2

### Constants ###   
DEMO_VIDEO = "corner.mp4"                         # The path to the demo video
MIN_NUM_MATCHES = 5                              # The minimum number of matches to be considered a match

class WallTraker:
    def __init__(self, initial_frame: np.array, total_interval: int, interval_length: int, skip_interval: int) -> None:
        self.total_interval = total_interval      # Total number of intervals in the demo video
        self.interval_length = interval_length    # Number of frames in a timeline interval
        self.skip_interval = skip_interval        # Interval between donkey and carrot
        self.accumulated_y_ratio = 1.0            # Accumulated y ratio
        self.total_states: List[State] = []       # A list of all states in the demo video
        self._load_all_states()                   # Load all frames from the demo video into a list of states
        self.robot_state = State(initial_frame)   # Create a state object for the robot
        self.donkey_index = -1                    # The index of the donkey state
        self.carrot_index = -1                    # The index of the carrot state
        self._find_donkey_carrot_state()          # Find the donkey and carrot state
        

    def __str__(self) -> str:
        pass

    def _load_all_states(self) -> None:
        '''
        description: Load all frames from the demo video into a list of states
        param       {*} self: -
        return      {*}: None
        '''
        # Create a VideoCapture object to read the video file
        video = cv2.VideoCapture(DEMO_VIDEO)
        for index in range(self.total_interval):
            print("Loading interval: " + str(index+1))
            # Read a frame from the video
            video.set(cv2.CAP_PROP_POS_FRAMES, (index+1)*self.interval_length)
            ret, frame = video.read()
            if not ret:
                raise Exception("Can't receive frame (stream end?). Exiting ...")
            # Create a state object
            state = State(frame, load=True, interval=index+1)
            self.total_states.append(state)
        video.release()


    def _find_donkey_carrot_state(self):
        """
        NOTE: This function will only be run once in the constructor
        description: Find the donkey and carrot state when looking sideways
        param       {*} self: -
        return      {*}: None
        """
        # Find the state with the least distance to the robot state
        min_distance = math.inf
        for index, state in enumerate(self.total_states):
            distance = self.robot_state.get_match_distance(state)
            if distance < min_distance:
                min_distance = distance
                self.donkey_index = index
        self.carrot_index = self.donkey_index + self.skip_interval
        print("donkey_index: ", self.donkey_index)
        if self.carrot_index >= self.total_interval:
            self.carrot_index = self.total_interval - 1
            print ("#-------- The donkey is too close to the destination --------#")
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
            # dynamic_gain = 1/(y_ratio_diff ** 1.7)
            dynamic_gain = 1/math.exp(y_ratio_diff) 
            # Calculate the new accumulated y ratio
            self.accumulated_y_ratio = self.accumulated_y_ratio * (1-dynamic_gain) + new_y_ratio * dynamic_gain
        return self.accumulated_y_ratio
    
    def chase_carrot(self)-> Tuple[int, float, int, bool]:
        '''
        description: Let robot chase the carrot
        param       {*} self: -
        return      {Tuple[int, float, bool]}: The linear velocity and angular velocity of the robot and whether the robot is close to the carrot
        '''
        query_coordinate, train_coordinate, num_matches = self.robot_state.get_match_coordinate(self.carrot_state)
        # If no match is found, return 0 velocity
        print("num_matches: ", num_matches)
        if num_matches <= MIN_NUM_MATCHES: return 0, 1, num_matches, True
        # Calculate the average x and y difference
        robot_center_x, robot_center_y, robot_x_radius, robot_y_radius = self.robot_state.compute_confidence_ellipse(query_coordinate)
        carrot_center_x, carrot_center_y, carrot_x_radius, carrot_y_radius = self.carrot_state.compute_confidence_ellipse(train_coordinate)
        robot_ellipse_ratio = robot_x_radius / robot_y_radius
        x_diff = robot_center_x - carrot_center_x
        y_ratio = robot_y_radius / carrot_y_radius
        # Calculate the moving average of the y ratio
        processed_y_ratio = self._calculate_moving_average_y(y_ratio)
        print("ellipse_ratio: ", robot_ellipse_ratio)
        return x_diff, processed_y_ratio, num_matches, False
    
    def next_carrot(self) -> int:
        '''
        description: Update the carrot state
        param       {*} self: -
        return      {*}: None
        '''
        self.carrot_index += self.skip_interval
        if self.carrot_index >= self.total_interval:
            self.carrot_index = self.total_interval
        else:
            self.carrot_state = self.total_states[self.carrot_index]
        return self.carrot_index

    def update_robot(self, new_frame: np.array) -> None:
        '''
        description: Update the robot state
        param       {*} self: -
        return      {*}: None
        '''
        self.robot_state = State(new_frame)

    def show_all_frames(self) -> Tuple[np.array, np.array]:
        '''
        description: Show all frames in the demo video
        param       {*} self: -
        return      {*}: None
        '''
        # Create windows
        cv2.namedWindow("Robot", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Donkey", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Carrot", cv2.WINDOW_NORMAL)

        # Move the windows to desired positions on the screen
        cv2.moveWindow("Robot", 0, 200)       # Position the "Robot" window at (100, 100)
        cv2.moveWindow("Donkey", 400, 200)      # Position the "Donkey" window at (600, 100)
        cv2.moveWindow("Carrot", 800, 200)     # Position the "Carrot" window at (1100, 100)

        robot = self.robot_state.temp_frame
        carrot = self.carrot_state.temp_frame

        # Display the frames in their respective windows
        self.robot_state.show_frame("Robot")
        self.donkey_state.show_frame("Donkey")
        self.carrot_state.show_frame("Carrot")
        # Debug
        # Show the carrot frame
        # cv2.imshow("Carrot_Original", self.carrot_state.frame)
        
        return robot, carrot