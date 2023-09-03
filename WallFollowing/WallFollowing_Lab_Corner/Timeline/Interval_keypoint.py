'''
Author       : Hanqing Qi
Date         : 2023-09-03 09:05:06
LastEditors  : Hanqing Qi
LastEditTime : 2023-09-03 10:13:02
FilePath     : /Timeline/Interval_keypoint.py
Description  : Class representing a keypoint
'''
#----- Import -----#
import numpy as np


class Interval_keypoint:
    def __init__(self, last_macth: int, starting_interval: int = None, ending_interval: int = None):
        self.starting_interval = starting_interval
        self.ending_interval = ending_interval
        self.last_macth = last_macth

    def get_last_match(self):
        return self.last_macth

    def update_last_match(self, last_match: int):
        self.last_macth = last_match
        self.ending_interval = self.ending_interval + 1