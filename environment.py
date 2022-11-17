import random
import math
from os.path import exists

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

        # load if it exists
        f = "discrete_data.npy"
        if exists(f):
            states_as_arrays = np.load(f) 
            print("Loaded file {0} with shape {1}.".format(f, states_as_arrays.shape))
            return states_as_arrays

        # else, build the states from scratch
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
        joint_lists = [np.asarray(v) for k, v in joint_values.items()]
        states_as_arrays = self.cartesian(joint_lists)

        # return unique subset of arrays
        # 41m in discrete case with 0.2 step
        states_as_arrays = np.unique(states_as_arrays, axis=0)
        np.save(f, states_as_arrays)
        print("Saved file {0} with shape {1}.".format(f, states_as_arrays.shape))
        return states_as_arrays


    def cartesian(self, arrays, out=None):
        # itertools.product() took too long and would freeze
        # source: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/11146645#11146645

        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = n / arrays[0].size
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out=out[0:m,1:])
            for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m,1:] = out[0:m,1:]
        return out


    def euclidean_distance(self, robot_pos):
        x, x2 = robot_pos[0], self.target_pos[0]
        y, y2 = robot_pos[1], self.target_pos[1]
        z, z2 = robot_pos[2], self.target_pos[2]
        return math.sqrt((z2 - z)**2 + (y2 - y)**2 + (x2 - x)**2)


    def get_reward(self, robot_pos):
        # return -self.euclidean_distance(robot_pos)
        return -1

    
    def reached_target(self, robot_pos):
        return robot_pos == self.target_pos


    # TODO
    # function to check whether the action is illegal (if it throws an exception
    #   such as if there is a collision created despite the joint values being
    #   within range

