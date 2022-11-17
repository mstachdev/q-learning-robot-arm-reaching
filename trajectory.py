import pickle

from agent import Agent
from environment import Environment
from qlearn import QLearning


# grab saved q table
with open('saved_q_table_lr0_925.pickle', 'rb') as handle:
    saved_q = pickle.load(handle)


# build agent, env, and qlearn objects
DELTA = 0.25
TARGET = (0.5,
          -0.5,
          -0.5,
          1,
          0,
          0)
EPSILON = 0.1
DISCOUNT_R = 0.9
LEARNING_R = 0.925
EPISODES_N = 5000
RUNS_N = 1

a = Agent(DELTA)
e = Environment(a, TARGET)
q = QLearning(epsilon=EPSILON,
              discount_rate=DISCOUNT_R,
              learning_rate=LEARNING_R,
              agent=a,
              environment=e,
              episodes=EPISODES_N,
              runs=RUNS_N)

# run optimal episode with saved q table
optimal_episode = q.run(saved_q)

# execute optimal episode as trajectory on Niryo Ned2
a.execute_trajectory(optimal_episode)


