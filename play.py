import tensorflow as tf

from config import Config
from dqn.agent import Agent
from dqn.environment import Environment

with tf.Session() as sess:
    config = Config()
    environment = Environment(config)
    agent = Agent(config, environment, sess)
    agent.play()
