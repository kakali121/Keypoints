'''
Author       : Karen Li
Date         : 2023-08-12 12:46:11
LastEditors  : Karen Li
LastEditTime : 2023-09-11 15:16:07
FilePath     : /WallFollowing_Corner_1/State.py
Description  : Class to represent a state
'''

### Import Packages ###
from typing import Tuple, List
import numpy as np
import cv2

### Constants ###
DESCRIPTOR_FILE_PATH = "side_demo_kpt_des_10_40"      # Path to the descriptor files
DESCRIPTOR_FILE = "side_demo_kpt_des"       # Name of the descriptor file

class State:
    def __init__(
        self, frame: np.array, max_humming_distance: int = 50, load: bool = False, interval: int = None
    ) -> None:
        self.frame = frame
        self.temp_frame = frame.copy() # A copy of the frame to be used for drawing
        self.keypoints = None
        self.descriptors = None
        self.MAX_HUMMING_DISTANCE = max_humming_distance
        if load: self._load_kpt_des(interval) # Load the keypoints and descriptors from the descriptor file
        else: self._extract_kpt_des() # Extract the keypoints and descriptors from the frame

    def __str__(self) -> str:
        pass

    def _load_kpt_des(self, interval: int) -> None:
        '''
        description: Load the keypoints and descriptors from the descriptor file
        param       {*} self: -
        param       {int} interval: The timeline interval of the state
        return      {*}: None
        '''
        file_name = (
            DESCRIPTOR_FILE_PATH + "/" + DESCRIPTOR_FILE + str(interval) + ".yml"
        )
        file_storage = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        # Load the keypoints and descriptors from the file
        self.descriptors = file_storage.getNode("descriptors").mat()
        self.keypoints = file_storage.getNode("keypoints").mat()
        file_storage.release()

    def _extract_kpt_des(self) -> None:
        '''
        description: Extract the keypoints and descriptors from the frame
        param       {*} self: -
        return      {*}: None
        '''
        # Create an ORB object and detect keypoints and descriptors
        ORB_Object = cv2.ORB_create(nfeatures=1000)
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # Detect keypoints and compute descriptors in the frame
        temp_keypoints, self.descriptors = ORB_Object.detectAndCompute(gray_frame, None)
        # Convert the keypoints to a numpy array
        self.keypoints = np.array([keypoint.pt for keypoint in temp_keypoints])
    
    def _find_match_pair(
        self, matches: List[np.array], state: 'State'
    ) -> Tuple[List[np.array], List[np.array]]:
        '''
        description: Find the coordinates of the matched keypoints
        param       {*} self: -
        param       {List} matches: The list of matches
        param       {*} state: The state to be compared with
        return      {Tuple[List[np.array], List[np.array]]}: The coordinates of the matched keypoints
        '''
        query_coordinte, train_coordinate = [], []
        for match in matches:
            query_index = match.queryIdx
            (x1, y1) = self.keypoints[query_index]
            query_coordinte.append(np.array((x1, y1)))
            train_index = match.trainIdx
            (x2, y2) = state.keypoints[train_index]
            train_coordinate.append(np.array((x2, y2)))
        return query_coordinte, train_coordinate

    def _get_matches(self, state: 'State') -> List[cv2.DMatch]:
        '''
        description: Get the matches between the state and another state
        param       {*} self: - 
        param       {State} state: The state to be compared with
        return      {*}: None
        '''
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return matcher.match(self.descriptors, state.descriptors)

    def get_match_coordinate(self, state: 'State', filter: bool = True, draw_keypoints: bool=True) -> Tuple[List[np.array], List[np.array]]:
        '''
        description: Get the coordinates of the matched keypoints
        param       {*} self: -
        param       {State} state: The state to be compared with 
        param       {bool} filter: Whether to filter the matches by distance
        return      {Tuple[List[np.array], List[np.array], int]}: The coordinates and number of the matched keypoints
        '''
        if not isinstance(state, State):
            raise TypeError("The input must be a State object")
        matches = self._get_matches(state)
        if matches is None: return None, None, 0
        if filter:
            matches = [match for match in matches if match.distance < self.MAX_HUMMING_DISTANCE]
        query_coordinte, train_coordinate = self._find_match_pair(matches, state)
        if draw_keypoints:
            self._draw_keypoints_pair(query_coordinte, train_coordinate)
            state._draw_keypoints_pair(train_coordinate, query_coordinte)
        return query_coordinte, train_coordinate, len(matches)

    def get_match_distance(self, state: 'State') -> float:
        '''
        description: Get the average distance of the matched keypoints
        param       {*} self: -
        param       {State} state: The state to be compared with
        return      {float}: The average distance of the matched keypoints
        '''
        if not isinstance(state, State):
            raise TypeError("The input must be a State object")
        matches = self._get_matches(state)
        if not matches: return float('inf')  # or some other value indicating no matches
        return sum(match.distance for match in matches) / len(matches)

    def _draw_keypoints_pair(self, query_coordinate: np.array, train_coordinate: np.array) -> None:
        '''
        description: Draw the keypoints and arrows between matched keypoints
        param       {*} self: -
        param       {np} query_coordinate: (x,y) coordinates of the query keypoints
        param       {np} train_coordinate: (x,y) coordinates of the train keypoints
        return      {*}: None
        '''
        # Validate input
        if len(query_coordinate) != len(train_coordinate):
            raise ValueError("The number of query coordinates must match the number of train coordinates.")

        # Draw keypoints and arrows
        for q_point, t_point in zip(query_coordinate, train_coordinate):
            q_point = (int(q_point[0]), int(q_point[1]))
            t_point = (int(t_point[0]), int(t_point[1]))

            # Draw query and train keypoints
            self.temp_frame = cv2.circle(self.temp_frame, q_point, 4, (0, 255, 0), -1)  # Green for query
            self.temp_frame = cv2.circle(self.temp_frame, t_point, 4, (0, 0, 255), -1)  # Red for train

            # Draw arrow between matched keypoints
            self.temp_frame = cv2.arrowedLine(self.temp_frame, q_point, t_point, (255, 0, 0), 2, tipLength=0.2)  # Blue for arrows

    def compute_confidence_ellipse(self, query_coordinate: np.array, n_std=2.5, draw: bool = True) -> Tuple[int, int, int, int]:
        '''
        description: Draw the confidence ellipse of the matched keypoints
        param       {*} self: -
        param       {np} query_coordinate: (x,y) coordinates of the query keypoints
        param       {*} n_std: standard deviation
        param       {bool} draw: Whether to draw the ellipse on the frame
        return      {*}: None
        '''
        query_coordinate_x = np.array([x for (x, y) in query_coordinate])
        query_coordinate_y = np.array([y for (x, y) in query_coordinate])

        # Compute the standard deviations of x and y
        std_x = np.std(query_coordinate_x) * n_std
        std_y = np.std(query_coordinate_y) * n_std

        mean_x = int(np.mean(query_coordinate_x))
        mean_y = int(np.mean(query_coordinate_y))

        if draw:
            # Draw the ellipse on the frame
            self.temp_frame = cv2.ellipse(self.temp_frame, (mean_x, mean_y), (int(std_x), int(std_y)), 0, 0, 360, (255, 0, 255), 2)
            # Draw the center of the ellipse
            self.temp_frame = cv2.circle(self.temp_frame, (mean_x, mean_y), 5, (255, 0, 255), -1)
            
        # Return the computed parameters (Optional, based on your needs)
        return mean_x, mean_y, std_x*2, std_y*2

    def show_frame(self, title: str) -> None:
        '''
        description: Show the frame after drawing
        param       {*} self: -
        return      {*}: None
        '''
        cv2.imshow(title, self.temp_frame)
        self.temp_frame = self.frame.copy() # Reset the temp frame