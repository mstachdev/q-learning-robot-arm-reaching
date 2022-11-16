import random

import numpy as np
from niryo_robot_python_ros_wrapper import *
import rospy

class Agent:
    def __init__(self, delta):
        self.s = (0, 0, 0, 0, 0, 0)
        self.a = set_discrete_action_set(delta)
        self.joint_ranges = {1 : (-2.967, 2.967),
                             2 : (-1.91, 0.61),
                             3 : (-1.34, 1.57),
                             4 : (-2.09, 2.09),
                             5 : (-1.92, 1.05),
                             6 : (-2.53, 2.53)}

    def reset_agent(self):
        self.s = (0, 0, 0, 0, 0, 0)


    def get_start(self):
        self.reset_agent()
        return self.s
        

    def set_discrete_action_set(delta):
        # 12 actions
        #   +/- delta to each of 6 joint values
        actions = []
        for i in range(6):
            a = [0]*6
            a2 = a.copy()
            a[i] = delta
            a2[i] = -delta
            actions.append(tuple(a))
            actions.append(tuple(a2))
        return actions

    
    def get_actions(self):
        return self.a

    
    def get_valid_actions(self, joint_values):
        # valid is defined as not exceeding
        #   the possible range of values for
        #   any of the 6 joints in Niryo Ned2.
        valid_actions = []
        for joint_adjustment in self.a:
            new_state = joint_values + joint_adjustment
            for i in len(range(new_state)):
                valid_range = self.joint_ranges[i+1]
                joint_value = new_state[i]
                if joint_value >= valid_range[0] and joint_value <= valid_range[1]:
                    valid_actions.append(joint_adjustment)
        return valid_actions


            

        
