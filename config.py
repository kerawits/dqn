class Config:
    def __init__(self):
        self.scale = 10000

        # self.game = 'PongDeterministic-v3'
        self.game = 'BreakoutDeterministic-v3'
        # self.game = 'MsPacmanDeterministic-v3'
        # self.game = 'SpaceInvadersDeterministic-v3'
        # self.game = 'FlappyBird-v0'

        self.create_checkpoints = True
        self.create_summaries = True
        # self.create_checkpoints = False
        # self.create_summaries = False

        self.checkpoints_path = 'checkpoints'
        self.summaries_path = 'summaries'
        # self.checkpoints_path = 'checkpoints'
        # self.summaries_path = 'summaries'

        self.screen_height = 84
        self.screen_width = 84

        # self.double_dqn = True
        self.double_dqn = False

        # self.observe_display = True
        # self.train_display = True
        self.observe_display = False
        self.train_display = False

        self.play_display = True
        self.skipframe_display = False

        self.final_training = 50000000

        self.min_reward = -1
        self.max_reward = 1

        self.model_save_frequency = self.scale * 5
        self.stats_print_frequency = self.scale * 5
        self.summary_frequency = self.scale

        self.minibatch_size = 32
        self.replay_memory_size = self.scale * 100

        self.agent_history_length = 4
        self.target_network_update_frequency = self.scale * 1

        self.discount_factor = 0.99

        # action repeat (2, 5) is built in for gym
        self.action_repeat = 1
        self.update_frequency = 4

        self.learning_rate = 0.00025
        self.min_squared_gradient = 0.01

        self.gradient_momentum = 0.95
        self.squared_gradient_momentum = self.gradient_momentum
        self.gradient_decay = self.gradient_momentum
        self.momentum = 0.0

        self.initial_exploration = 1.0
        self.final_exploration = 0.1
        self.no_exploration = 0.05

        self.final_exploration_frame = self.scale * 100
        self.replay_start_size = self.scale * 5

        self.no_op_max = 5
