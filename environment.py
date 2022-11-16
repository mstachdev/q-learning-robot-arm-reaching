import random
import numpy as np

from niryo_robot_python_ros_wrapper import *
import rospy

class Environment:
    def __init__(self, target_pos):
        self.target_pos = target_pos # (x, y, z, 0, 0, 0) of target object


    # function to return reward

    # function to calculate distance between position of end effector and target

    # function to check whether the action is illegal (if it throws an exception
    #   such as if there is a collision created despite the joint values being
    #   within range

    # function to note starting point (fixed to start, then randomized)

    # function to check whether at target object / finished

    
    def get_start(self):
        # TODO

    def finished(self, x, y):
        # TODO
    
    def get_reward(self, x, y):
        # TODO
