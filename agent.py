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
        self.ned.move_joints(*self.s)

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

    
    def get_valid_actions(self, state):
        # valid is defined as not exceeding
        #   the possible range of values for
        #   any of the 6 joints in Niryo Ned2.
        valid_actions = []
        # 12 actions
        for action in self.a:
            new_state = tuple(np.asarray(state) + np.asarray(action))
            # check each component of new state (i.e., each joint value)
            #   is within its perscribed range
            update = True
            for i in range(len(new_state)):
                valid_range = self.joint_ranges[i+1]
                joint_value = new_state[i]
                # append the joint adjustment (i.e., action) if within both - and + ranges
                if (joint_value < valid_range[0]) or (joint_value > valid_range[1]):
                   update = False 
            # only update if new state is o.k. for every joint in 6-tuple
            if update:
                valid_actions.append(action)
        return valid_actions
    

    def execute_trajectory(self, episode, load=False, trajectory=None):
        # episode has form episode[t] = ((s, a), r)
        #   train using qlearn()
        #   then run an episode with qlearn.run()
        #   then transform that to extract the states (to get the trajectory)

        # load (and do not regenerate trajectory) if 'load' is set to True with
        #   a provided trajectory name to load
        if load:
            trajectory = self.get_saved_trajectory(trajectory)
            self.ned.execute_trajectory_from_poses_and_joints(trajectory)
            return 
        
        # a trajectory will contain the sequence of states (i.e., joint values) that
        #   we progressively move through
        trajectory = [tup[0][0] for tup in episode.values()]
        
        print("Running optimal episode with trajectory that has {0} steps; \n trajectory: {1}".format(len(trajectory), trajectory))
        
        # execute tracjectory using niryo's provided wrapper
        self.ned.execute_trajectory_from_poses_and_joints(trajectory, list_type=['joint'])

        # save trajectory
        self.ned.save_trajectory("ned_discrete_reaching_trajectory", trajectory)


