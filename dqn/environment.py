import gym
# import gym_ple
import random
import numpy as np
# from scipy.misc import imresize
import cv2

class Environment:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(self.config.game)

        print('game: {}'.format(self.config.game))
        print('action_repeat: {}'.format(self.config.action_repeat))
        print('number_of_actions: {}'.format(self.number_of_actions()))
        print('action_meanings: {}'.format(self.action_meanings()))

        try:
            print('env.frameskip: {}'.format(self.env.frameskip))
        except:
            pass

    def cap_reward(self, reward):
        return reward if reward == 0.0 else self.config.min_reward if reward < 0.0 else self.config.max_reward

    def new_game(self):
        self.env.reset()

        self.reward_game = 0
        self.score_game = 0
        self.lives = 0
        self.terminal = False

        s_t, r_t, self.terminal, info = self.env.step(0)
        try:
            self.lives = info['ale.lives']
        except:
            pass

    def game_over(self):
        return self.terminal and self.lives == 0

    def action(self, action, repeat=1, display=True):
        action_reward = 0
        for _ in range(repeat):
            x_tp, r_t, self.terminal, info = self.env.step(action)

            if display and self.config.skipframe_display:
                self.env.render()

            self.score_game += r_t
            action_reward += r_t

            if self.lives != 0:
                 if self.lives != info['ale.lives']:
                    action_reward = self.config.min_reward
                    self.terminal = True
                    self.lives = info['ale.lives']
                    break

            if self.terminal:
                break

        if display and  not self.config.skipframe_display:
            self.env.render()

        self.reward_game += self.cap_reward(action_reward)

        # print('action: {}, reward: {}, terminal: {}'.format(action, self.cap_reward(action_reward), self.terminal))

        return self.preprocess_image(x_tp), self.cap_reward(action_reward), self.terminal

    def preprocess_image(self, image):
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dsize=(self.config.screen_height, self.config.screen_width)) / 255

    def number_of_actions(self):
        return self.env.action_space.n

    def number_of_lives(self):
        return self.lives

    def action_meanings(self):
        try:
            return self.env.get_action_meanings()
        except:
            return []
