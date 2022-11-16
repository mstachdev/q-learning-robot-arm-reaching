import random
import math
import numpy as np

from niryo_robot_python_ros_wrapper import *
import rospy

class Environment:
    def __init__(self, agent, target_pos):
        self.agent = agent
        self.target_pos = target_pos # (x, y, z, 0, 0, 0) of target object
        self.start = self.agent.s

    # TODO
    # function to check whether the action is illegal (if it throws an exception
    #   such as if there is a collision created despite the joint values being
    #   within range

    def euclidean_distance(robot_pos):
        x, x2 = robot_pos[0], self.target_pos[0]
        y, y2 = robot_pos[1], self.target_pos[1]
        z, z2 = robot_pos[2], self.target_pos[2]
        return math.sqrt((z2 - z)**2 + (y2 - y)**2 + (x2 - x)**2)


    def get_reward(self, robot_pos):
        return - self.euclidean_distance(robot_pos)

    
    def reached_target(self, robot_pos, threshold=0.4):
        # default threshold value of 0.4 means about 0.2 deviation
        #   is permitted for each of x, y, z
        if self.euclidean_distance(robot_pos) < threshold:
            return True
        return False
