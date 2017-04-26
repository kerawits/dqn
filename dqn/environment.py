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
        # print(repeat)
        reward = 0
        terminal_tp = False
        for _ in range(repeat):
            x_tp, r_t, terminal_tp, info = self.env.step(action)
            self.score_game += r_t

            try:
                # check if a live was lost in a multi-live games
                if self.lives != 0 and self.lives != info['ale.lives']:
                    terminal_tp = True
                    r_t += self.config.min_reward
                    self.lives = info['ale.lives']
            except:
                # do nothing
                pass

            reward += self.cap_reward(r_t)

            if terminal_tp:
                self.terminal = True
                break

        self.reward_game += self.cap_reward(reward)

        if display:
            self.env.render()

        # if reward != 0:
        #     print('========reward========')
        #     print('reward: {}'.format(reward))
        #     print('terminal: {}'.format(terminal_tp))
        #     print('self.lives: {}'.format(self.lives))
        #     print('self.reward_game: {}'.format(self.reward_game))
        #     print('self.score_game: {}'.format(self.score_game))
        #     print('game_over: {}'.format(self.game_over()))

        return self.preprocess_image(x_tp), self.cap_reward(reward), self.terminal

    # def rgb2gray(self, image):
    #     return np.dot(image[...,:3] , [0.299, 0.587, 0.114])

    def preprocess_image(self, image):
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dsize=(self.config.screen_height, self.config.screen_width))

    def number_of_actions(self):
        return self.env.action_space.n

    def number_of_lives(self):
        return self.lives

    def action_meanings(self):
        try:
            return self.env.get_action_meanings()
        except:
            return []
