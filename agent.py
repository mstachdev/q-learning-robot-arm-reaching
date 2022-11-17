import random

import numpy as np
from niryo_robot_python_ros_wrapper import *
import rospy

class Agent:
    def __init__(self, delta):
        self.delta = delta
        self.s = (0, 0, 0, 0, 0, 0)
        self.a = self.set_discrete_action_space(self.delta)
        self.joint_ranges = {1 : (-1.25, 1.25),
                             2 : (-1.25, 0.25),
                             3 : (-1, 1),
                             4 : (-1.75, 1.75),
                             5 : (-0.25, .25),
                             6 : (-0.25, 0.25)}
        self.n_joints = len(self.joint_ranges)

        # initialize node
        rospy.init_node('ned2_discrete_reaching')

        # store wrapper
        self.ned = NiryoRosWrapper()

    def reset_agent(self):
        self.s = (0, 0, 0, 0, 0, 0)


    def get_start(self):
        # reset ned to starting position 
        self.reset_agent()

        # move ned to starting position
        self.ned.move_joints(self.s)

        return self.s
        

    def set_discrete_action_space(self, delta):
        # 12 actions
        #   +/- delta to each of 6 joint values
        actions = []
        for i in range(6):
            a = [0]*6
            a2 = list(a) # create a local copy
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
            # check each component of new state (i.e., each joint value)
            #   is within its perscribed range
            for i in len(range(new_state)):
                valid_range = self.joint_ranges[i+1]
                joint_value = new_state[i]
                # append the joint adjustment (i.e., action) if within both - and + ranges
                if joint_value >= valid_range[0] and joint_value <= valid_range[1]:
                    valid_actions.append(joint_adjustment)
        return valid_actions
    

    def execute_trajectory(self, episode):
        # episode has form episode[t] = ((s, a), r)
        
        # a trajectory will contain the sequence of states (i.e., joint values) that
        #   we progressively move through
        joint_value_lists = [s[0][0] for tup in episode]
        
        # execute tracjectory using niryo's provided wrapper
        self.ned.execute_trajectory_from_poses_and_joints(joint_value_lists)


