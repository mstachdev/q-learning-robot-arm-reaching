import random
import math
from itertools import product

import numpy as np
from niryo_robot_python_ros_wrapper import *
import rospy

class Environment:
    def __init__(self, agent, target_pos):
        self.agent = agent
        self.target_pos = target_pos # (x, y, z, 0, 0, 0) of target object
        self.start = self.agent.s
        self.state_space = self.set_discrete_state_space(self.agent.delta)


    def set_discrete_state_space(self, delta):
        # delta is the value (e.g., 0.20) that we increment/decrement
        #   the joint values by in our actions
        #
        # this function returns a list of all possible states in the
        #   discrete case. these are calculated by generating every 
        #   combination working outward in the +/- directions by delta
        #   from the starting state's value for each joint value
        #
        # the algorithm first generates the 6 lists of possible joint values
        #   so 1 list per joint. it then takes the cross product and removes
        #   duplicates so the final product returned is the set of all
        #   combinations of valid (in terms of ranges of joint values) 
        #   joint values that are our states in the discrete case.

        start = self.agent.s # need to create states, incrementing/decrementing
                             # from the start to ensure these states are reachable

        # get all possible joint values by joint
        joint_values = { i+1 : [] for i in range(self.agent.n_joints) } 
        for i in range(6):
            values = []
            joint_range = self.agent.joint_ranges[i+1] # get valid range of joint values
            left_ptr = self.start[i] # get corresponding joint value from start
            right_ptr = left_ptr
            while(left_ptr >= joint_range[0] or right_ptr <= joint_range[1]):
                if left_ptr >= joint_range[0]:
                    values.append(left_ptr)
                if right_ptr <= joint_range[1]:
                    values.append(right_ptr)
                left_ptr -= delta
                right_ptr += delta
            joint_values[i+1].extend(values)

        # cross product these lists of joint values to get all possible states
        joint_lists = [v for k, v in joint_values.items()]
        states = set(product(*joint_lists))

        return states


    def euclidean_distance(self, robot_pos):
        x, x2 = robot_pos[0], self.target_pos[0]
        y, y2 = robot_pos[1], self.target_pos[1]
        z, z2 = robot_pos[2], self.target_pos[2]
        return math.sqrt((z2 - z)**2 + (y2 - y)**2 + (x2 - x)**2)


    def get_reward(self, robot_pos):
        return -self.euclidean_distance(robot_pos)

    
    def reached_target(self, robot_pos, threshold=0.5):
        # default threshold value of 0.4 means about 0.2-0.3 deviation
        #   is permitted for each of x, y, z
        if self.euclidean_distance(robot_pos) < threshold:
            return True
        return False


    # TODO
    # function to check whether the action is illegal (if it throws an exception
    #   such as if there is a collision created despite the joint values being
    #   within range

