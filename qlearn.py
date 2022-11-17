import random

import numpy as np
from niryo_robot_python_ros_wrapper import *
import rospy

class QLearning:
    def __init__(self, epsilon, discount_rate, learning_rate, agent, environment, episodes, runs):
        self.e = epsilon
        self.d = discount_rate
        self.l = learning_rate
        self.env = environment
        self.agent = agent
        self.runs = runs
        self.episodes = episodes
        self.episode = {}           # e[t] = ((s, a), r)
        self.Q = self.initialize()  # Q[sa] = 0s to start
    

    def initialize(self):
        # self.agent.a holds the 12 discrete actions
        # self.env.state_space holds all the possible states, in terms of joint values
        #
        # by default this takes each state and creates 12 versions of it, each with a
        #   possible action value
        q = {}
        for state in self.env.state_space: # each state is a numpy array
            for action in self.agent.a: # each action is a tuple
                s = tuple(state.tolist())
                sa = s + action # tuple state-action pair
                q[sa] = 0
        return q


    def get_next_action(self, state):
        # state has form 6-tuple with joint values
        if(np.random.binomial(1, self.e)): # 1 if exploring, 0 if exploiting
            valid_actions = self.agent.get_valid_actions(state) # pass the joint values 
            exploring = valid_actions[random.randint(0, len(valid_actions))-1]
            return exploring # e.g., (0,0,0,-0.25,0,0) or an adjustment to a single joint
        a = self.get_greedy_action(state)
        return a
     
 
    def get_greedy_action(self, state):
        # state has form 6-tuple with joint values
        # state, action as 1 tuple of 12 values - a 12-tuple

        valid_actions = self.agent.get_valid_actions(state)
        if len(valid_actions)==0:
            print("No valid actions")
        sas = [state + a for a in valid_actions] 
        subset_dict = {sa:self.Q[sa] for sa in sas} # subset dict to keys

        # random, because subset_dict is empty
        if len(subset_dict)==0:
            print("empty dict")
            return valid_actions[random.randint(0, len(valid_actions))-1]
        # greedy
        idxs = max(subset_dict, key=subset_dict.get) # return key for max value
        return idxs[-6:] # return (dx_j1, dx_j2, dx_j3, dx_j4, dx_j5, dx_j6) 
     
     
    def update_state(self, state, action):
        # state has form 6-tuple of joint values
        # action likewise has form 6-tuple, and it has the single joint adjustment

        # update state
        state_array = np.asarray(state)
        action_array = np.asarray(action)
        new_state = state_array + action_array

        return tuple(new_state) # return new state as 6-tuple
 
 
    def update_q(self, old_state, old_action, new_state, new_action, r):
        # all inputs except r are tuples

        old_sa = old_state + old_action
        new_sa = new_state + new_action
        # set target
        target = r + (self.d * self.Q[new_sa])
        # make on policy update
        #   it is on policy because the key (state, action pair) where we update the q function
        #   is the same as the one that generated the action
        oldq = self.Q[old_sa]
        self.Q[old_sa] = oldq + self.l*(target - oldq)
     

    def DEBUG_PRINT(self, timestep, old_s, old_a, s, a):
        # [0] print time step
        print(timestep)

        # [1] details about state-action update
        print("{0} and {1} -> {2} and {3}".format(old_s, old_a, s, a))

        # [2] details about action
        #   whether greedy or exploratory
        #   what q key-value pairs it was selected from
        sa_pairs = [tuple(old_s + a) for a in self.agent.get_valid_actions(old_s)]
        qvalues = [(sa, self.Q[sa]) for sa in sa_pairs]
        print("{0} action selected was from {1}".format(a, qvalues))

        # new line to indicate new episode
        print()


    def run(self, q=None):
        # use any provided q table
        if q:
            self.Q = q
        self.episode = {}
        t = 0
        r = 0
        tr = r
        s = self.agent.get_start() 
        a = self.get_next_action(s)
        while(not self.env.reached_target(s)):
            old_s, old_a = s, a
            s = self.update_state(old_s, old_a)
            r = self.env.get_reward(s)
            a = self.get_next_action(s)
            greedy_a = self.get_greedy_action(s)
            self.update_q(old_s, old_a, s, greedy_a, r) 
            self.episode[t] = ((s, a), r)
            t += 1
            tr += r
        print("Reached target in {0} time steps with {1} total reward, ending at state {2}.".format(t, tr, s))
        self.episode[t] = ((s, a), r)
        return self.episode
     
 
    def __call__(self):
        episodes_in_run = []
        for r in range(self.runs): # 50
            self.Q = self.initialize() 
            cr_by_episode = []

            # The difference between the main loop in q-learning and in sarsa is
            #   in how the update to the action value (q) is made. In sarsa, it 
            #   is made using s', a' using the same policy; in q-learning, we
            #   make the update using not s', a' but the state-action pair that
            #   has the highest q value so far - i.e., we follow a different 
            #   policy where we are greedy.
            for e in range(self.episodes): # 500
                print(e)
                self.episode = {}
                t = 0
                r = 0
                tr = r
                s = self.agent.get_start() 
                a = self.get_next_action(s)
                while(not self.env.reached_target(s)):
                    old_s, old_a = s, a
                    s = self.update_state(old_s, old_a)
                    r = self.env.get_reward(s)

                    a = self.get_next_action(s)

                    # we pass the greedy action (greedy_dx, greedy_dy) to update_q() rather than
                    # the policy's dx, dy we get above

                    greedy_a = self.get_greedy_action(s)
                    
                    self.update_q(old_s, old_a, s, greedy_a, r) 

                    # store episode
                    #  the last episode will be used for the trajectory to move the robot
                    self.episode[t] = ((s, a), r)
                    # self.DEBUG_PRINT(t, old_s, old_a, s, a)
                    t += 1
                    tr += r

                print("Reached target in {0} time steps with {1} total reward, ending at state {2}.".format(t, tr, s))
                cr_by_episode.append([tr])
                self.episode[t] = ((s, a), r)
            episodes_in_run.append(cr_by_episode) 
        return episodes_in_run

