import numpy as np
import random
import os

class Memory:
    def __init__(self, config):
        self.config = config

        self.actions = np.empty(self.config.replay_memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.config.replay_memory_size, dtype=np.integer)
        self.screens = np.empty((self.config.replay_memory_size, self.config.screen_height, self.config.screen_width), dtype=np.float32)
        self.terminals = np.empty(self.config.replay_memory_size, dtype=np.bool)

        self.count = 0
        self.current = 0

        # NCHW
        self.pre_states = np.empty((self.config.minibatch_size, self.config.agent_history_length, self.config.screen_height, self.config.screen_width), dtype=np.float32)
        self.pos_states = np.empty((self.config.minibatch_size, self.config.agent_history_length, self.config.screen_height, self.config.screen_width), dtype=np.float32)

    def add(self, x_tp, r_t, a_t, terminal_tp):
        self.screens[self.current, ...] = x_tp
        self.rewards[self.current] = r_t
        self.actions[self.current] = a_t
        self.terminals[self.current] = terminal_tp

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.replay_memory_size

    def getState(self, index):
        index = index % self.count

        if index >= self.config.agent_history_length - 1:
            state = self.screens[(index - (self.config.agent_history_length - 1)):(index + 1), ...]
        else:
            indexes = [(index - i) % self.count for i in reversed(range(self.config.agent_history_length))]
            state = self.screens[indexes, ...]

        return state

    def sample(self):
        indexes = []
        while len(indexes) < self.config.minibatch_size:
            index = random.randint(self.config.agent_history_length, self.count - 1)

            if index >= self.current and index < self.current + self.config.agent_history_length or self.terminals[(index - self.config.agent_history_length):index].any():
                continue
            else:
                # print('index({}): {}, a_t: {}'.format(len(indexes), index, self.actions[index]))
                indexes.append(index)

        # print('indexes: ', indexes)
        for i, index in enumerate(indexes):
            self.pre_states[i, ...] = self.getState(index - 1)
            self.pos_states[i, ...] = self.getState(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return np.transpose(self.pre_states, [0, 2, 3, 1]), actions, rewards, np.transpose(self.pos_states, [0, 2, 3, 1]), terminals

    def save(self):
        print('Saving Replay Memory...')
        for idx, (name, array) in enumerate(
            zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
                [self.actions, self.rewards, self.screens, self.terminals, self.pre_states, self.pos_states])):
            try:
                self.save_npy(array, os.path.join(self.config.checkpoints_path, self.config.game + '-' + name))
            except:
                print('Failed: {}'.format(os.path.join(self.config.checkpoints_path, self.config.game + '-' + name)))

    def load(self):
        print('Loading Replay Memory...')
        for idx, (name, array) in enumerate(
            zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
                [self.actions, self.rewards, self.screens, self.terminals, self.pre_states, self.pos_states])):
            try:
                array = self.load_npy(os.path.join(self.config.checkpoints_path, self.config.game + '-' + name + '.npy'))
            except:
                print('Failed: {}'.format(os.path.join(self.config.checkpoints_path, self.config.game + '-' + name + '.npy')))

    def save_npy(self, obj, path):
        np.save(path, obj)
        print('Replay Memory saved: {}'.format(path))

    def load_npy(self, path):
        obj = np.load(path)
        print('Replay MemoryLoaded: {}'.format(path))
        return obj
