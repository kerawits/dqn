import numpy as np
# np.set_printoptions(threshold=np.inf)

class History:
    def __init__(self, config):
        self.config = config

        self.agent_history_length, self.screen_height, self.screen_width = config.agent_history_length, config.screen_height, config.screen_width
        self.history = np.zeros([self.agent_history_length, self.screen_height, self.screen_width], dtype=np.float32)

    def add(self, screen):
        # print('screen min: {} max: {} avg: {}'.format(np.min(screen), np.max(screen), np.mean(screen)))
        # print('screen {}'.format(screen))

        # shift left and replace the last index with the input
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        self.history = np.zeros([self.agent_history_length, self.screen_height, self.screen_width], dtype=np.float32)

    def get(self):
        # print('Historypredict: np.transpose(self.history, (1, 2, 0)): {}: {}'.format(np.transpose(self.history, (1, 2, 0)).dtype, np.transpose(self.history, (1, 2, 0)).shape))

        # print('self.history min: {} max: {} avg: {}'.format(np.min(self.history), np.max(self.history), np.mean(self.history)))
        # print('self.history {}'.format(self.history))

        # CHW - > HWC
        return np.transpose(self.history, (1, 2, 0))
