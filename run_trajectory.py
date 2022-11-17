import pickle

from agent import Agent
from environment import Environment
from qlearn import QLearning


# grab saved q table
with open('optimal_episode26.pickle', 'rb') as handle:
    optimal_episode = pickle.load(handle)


# build agent
DELTA = 0.25
a = Agent(DELTA)

# execute optimal episode as trajectory on Niryo Ned2
a.execute_trajectory(optimal_episode)


