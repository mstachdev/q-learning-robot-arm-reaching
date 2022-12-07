'''
Script to test a single move on Niryo Ned2.

Intended to ensure connection to Niryo Ned2 before running more complex code.
'''
from niryo_robot_python_ros_wrapper import *

from agent import Agent

# single move of joints to arbitrary position
agent = agent(0.5)
agent.move_joints(0.1, -0.2, 0.0, 1.1, -0.5, 0.2)

