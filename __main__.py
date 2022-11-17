'''
Phase 1
Testing discrete state and action set with q-learning
Author: Matthew Stachyra
'''
from agent import Agent
from environment import Environment
from qlearn import QLearning

if __name__ == "__main__":
    DELTA = 0.25
    TARGET = (0.5,
              -0.5,
              -0.5,
              1,
              0,
              0)
    EPSILON = 0.1
    DISCOUNT_R = 0.9
    LEARNING_R = 0.9
    EPISODES_N = 250
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
    
    run_data = q()
    last_episode = q.episode
    print(last_episode)

