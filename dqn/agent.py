import numpy as np
import tensorflow as tf
import random
import os
import time
import datetime

from .history import History
from .memory import Memory
from .ops import *

import matplotlib.pyplot as plt

class Agent:
    def __init__(self, config, environment, sess):
        self.config = config

        self.sess = sess
        self.environment = environment
        self.history = History(self.config)
        self.memory = Memory(self.config)

        self.epsilon = self.config.initial_exploration
        self.step = 1
        self.episode = 0
        self.start_time = time.time()

        print('config: {}'.format(vars(config)))

        self.init_summary()
        self.init_statistics()
        self.buid_dqn()

    def predict(self, s_t, epsilon=None):
        # anneal epsilion linearly
        self.epsilon = epsilon or (self.config.initial_exploration - (self.config.initial_exploration - self.config.final_exploration) * min(self.step, self.config.final_exploration_frame) / (self.config.final_exploration_frame))

        if np.random.uniform() < self.epsilon:
            return random.randint(0, self.environment.number_of_actions() - 1)
        else:
            [q_action, Q] = self.sess.run([self.q_action, self.q], feed_dict={self.s_t: [s_t]})

            self.predicted_Q_max = np.append(self.predicted_Q_max, np.max(Q[0]))
            self.predicted_actions = np.append(self.predicted_actions, q_action[0])

            # print('Q: {}'.format(Q))

            return q_action[0]

    def init_summary(self):
        self.predicted_Q_max = np.array([])
        self.predicted_actions = np.array([])
        self.game_rewards = np.array([])
        self.game_scores = np.array([])

    def init_statistics(self):
        self.statistics_start_time = time.time()

    def print_statistics(self, step, stats_print_frequency):
        time_interval = time.time() -  self.statistics_start_time
        time_total = time.time() -  self.start_time
        print_with_time('step: {} ({})'.format(step, self.program_phase))

        print('time\t\t: {} Hr\tsteps/s\t\t: {: .4f}'.format(str(datetime.timedelta(seconds=time_total)), stats_print_frequency / time_interval))
        print('episode\t\t: {}\t\t\tepsilon\t\t: {: .4f}'.format(self.episode, self.epsilon))

    def init_history(self, x_tp):
        for _ in range(self.config.agent_history_length):
            self.observe(x_tp, 0, 0, False)

    def observe(self, x_tp, r_t, a_t, terminal_tp):
        self.history.add(x_tp)
        self.memory.add(x_tp, r_t, a_t, terminal_tp)

    def buid_dqn(self):
        with tf.variable_scope('step'):
            self.ckpt_step_placeholder = tf.placeholder(tf.float32, shape=[], name='ckpt_step_placeholder')
            self.ckpt_step = tf.get_variable('ckpt_step', shape=[])
            self.assign_to_ckpt_step_operation = self.ckpt_step.assign(self.ckpt_step_placeholder)

        # online network
        with tf.variable_scope('online'):
            self.s_t = tf.placeholder(tf.float32, shape=[None, self.config.screen_height, self.config.screen_width, self.config.agent_history_length], name='s_t')

            out1, self.w1, self.b1 = conv_relu(self.s_t, [8, 8], 4, 4, 32, 'VALID', 'conv1')
            out2, self.w2, self.b2 = conv_relu(out1, [4, 4], 2, 32, 64, 'VALID', 'conv2')
            out3, self.w3, self.b3 = conv_relu(out2, [3, 3], 1, 64, 64, 'VALID', 'conv3')
            out3_flat = tf.reshape(out3, [-1, 7 * 7 * 64])
            out4, self.w4, self.b4 = fc_relu(out3_flat, 7 * 7 * 64, 512, 'fc1')

            self.q, self.w5, self.b5 = fc_linear(out4, 512, self.environment.number_of_actions(), 'fc2')
            self.q_action = tf.argmax(self.q, axis=1)

            # tf.summary.image('sample_1', tf.reshape(tf.transpose(self.s_t[0], perm=[2, 0, 1]), [-1, 84, 84, 1]), max_outputs=4)
            # tf.summary.image('sample_2', tf.reshape(tf.transpose(self.s_t[1], perm=[2, 0, 1]), [-1, 84, 84, 1]), max_outputs=4)
            # tf.summary.image('sample_3', tf.reshape(tf.transpose(self.s_t[2], perm=[2, 0, 1]), [-1, 84, 84, 1]), max_outputs=4)
            # tf.summary.image('sample_4', tf.reshape(tf.transpose(self.s_t[3], perm=[2, 0, 1]), [-1, 84, 84, 1]), max_outputs=4)

        # target network
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder(tf.float32, [None, self.config.screen_height, self.config.screen_width, self.config.agent_history_length], name='target_s_t')

            target_out1, self.target_w1, self.target_b1 = conv_relu(self.target_s_t, [8, 8], 4, 4, 32, 'VALID', 'conv1')
            target_out2, self.target_w2, self.target_b2 = conv_relu(target_out1, [4, 4], 2, 32, 64, 'VALID', 'conv2')
            target_out3, self.target_w3, self.target_b3 = conv_relu(target_out2, [3, 3], 1, 64, 64,'VALID', 'cov3')
            target_out3_flat = tf.reshape(target_out3, [-1, 7 * 7 * 64])
            target_out4, self.target_w4, self.target_b4 = fc_relu(target_out3_flat, 7 * 7 * 64, 512, 'fc1')

            self.target_q, self.target_w5, self.target_b5 = fc_linear(target_out4, 512, self.environment.number_of_actions(), 'fc2')
            self.target_q_max = tf.reduce_max(self.target_q, axis=1)

            self.target_q_action = tf.placeholder(tf.int32, [None, None], 'target_q_action')
            self.target_q_with_q_action = tf.gather_nd(self.target_q, self.target_q_action, 'target_q_with_q_action')

        with tf.variable_scope('update_target_network'):
            self.assign_to_target_w1_operation = self.target_w1.assign(self.w1)
            self.assign_to_target_w2_operation = self.target_w2.assign(self.w2)
            self.assign_to_target_w3_operation = self.target_w3.assign(self.w3)
            self.assign_to_target_w4_operation = self.target_w4.assign(self.w4)
            self.assign_to_target_w5_operation = self.target_w5.assign(self.w5)

            self.assign_to_target_b1_operation = self.target_b1.assign(self.b1)
            self.assign_to_target_b2_operation = self.target_b2.assign(self.b2)
            self.assign_to_target_b3_operation = self.target_b3.assign(self.b3)
            self.assign_to_target_b4_operation = self.target_b4.assign(self.b4)
            self.assign_to_target_b5_operation = self.target_b5.assign(self.b5)

        with tf.variable_scope('optimizer'):
            self.q_t_target = tf.placeholder(tf.float32, [None], name='q_t_target')
            self.a_t = tf.placeholder(tf.uint8, [None], name='a_t')

            a_t_one_hot = tf.one_hot(self.a_t, self.environment.number_of_actions(), name='a_t_one_hot')
            q_t_acted = tf.reduce_sum(self.q * a_t_one_hot, reduction_indices=1, name='q_t_acted')

            # self.loss = tf.reduce_mean(huber_loss(q_t_acted - self.q_t_target, 1.0), name='loss')
            self.loss = tf.reduce_sum(huber_loss(q_t_acted - self.q_t_target, 1.0), name='loss')

            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate, momentum=self.config.momentum, epsilon=self.config.min_squared_gradient, name='RMSProp').minimize(self.loss)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate, decay=self.config.gradient_decay, momentum=self.config.momentum, epsilon=self.config.min_squared_gradient, centered=True, name='RMSProp').minimize(self.loss)

        with tf.variable_scope('performance'):
            self.actions_summary = tf.placeholder(tf.float32, [None], name='actions_summary')

            self.Q_max_summary = tf.placeholder(tf.float32, [None], name='Q_max_summary')
            self.rewards_summary = tf.placeholder(tf.float32, [None], name='rewards_summary')
            self.scores_summary = tf.placeholder(tf.float32, [None], name='scores_summary')

            with tf.variable_scope('actions'):
                tf.summary.histogram('histogram', self.actions_summary)

            if self.config.create_summaries:
                variable_summaries(self.Q_max_summary, name='Q_max')
                variable_summaries(self.rewards_summary, name='rewards')
                variable_summaries(self.scores_summary, name='scores')

        self.sess.run(tf.global_variables_initializer())

        if self.config.create_summaries:
            self.summary = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(os.path.join(self.config.summaries_path, self.config.game))
            self.summary_writer.add_graph(self.sess.graph)

        self.saver = tf.train.Saver(max_to_keep=0)
        self.load_model()

    def update_target_network(self):
        self.sess.run([self.assign_to_target_w1_operation, self.assign_to_target_w2_operation, self.assign_to_target_w3_operation, self.assign_to_target_w4_operation, self.assign_to_target_w5_operation, self.assign_to_target_b1_operation, self.assign_to_target_b2_operation, self.assign_to_target_b3_operation, self.assign_to_target_b4_operation, self.assign_to_target_b5_operation])

    def minibatch(self, double_dqn):
        s_t, a_t, r_t, s_tp, terminal_tp = self.memory.sample()

        terminal_tp = terminal_tp * 1.0

        q_tp_max = np.zeros([self.config.minibatch_size])
        q_t_target = np.zeros([self.config.minibatch_size])

        if double_dqn:
            q_tp_action = self.sess.run(self.q_action, feed_dict={self.s_t: s_tp})
            q_tp_max = self.sess.run(self.target_q_with_q_action, feed_dict={
                self.target_s_t: s_tp,
                self.target_q_action: [[i, action] for i, action in enumerate(q_tp_action)]
            })
        else:
            q_tp_max = self.sess.run(self.target_q_max, feed_dict={self.target_s_t: s_tp})

        q_t_target = (1 - terminal_tp) * self.config.discount_factor * q_tp_max + r_t

        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={
            self.q_t_target: q_t_target,
            self.s_t: s_t,
            self.a_t: a_t
        })

    def load_model(self):
        print_with_time('Loading checkpoints...')
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.config.checkpoints_path, self.config.game))
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(os.path.join(self.config.checkpoints_path, self.config.game), ckpt_name)
            self.saver.restore(self.sess, fname)
            self.step = int(self.ckpt_step.eval()) + 1
            print('Loaded: {}'.format(fname))
            return True
        else:
            print('Initializing networks with random weights...')
            self.step = 1
            return False

    def update_ckpt_step(self, step):
        self.sess.run([self.assign_to_ckpt_step_operation], feed_dict={
            self.ckpt_step_placeholder: step
        })

    def save_model(self, step):
        print_with_time('Saving checkpoints...')
        if not os.path.exists(os.path.join(self.config.checkpoints_path, self.config.game)):
            os.makedirs(os.path.join(self.config.checkpoints_path, self.config.game))
        self.saver.save(self.sess, os.path.join(os.path.join(self.config.checkpoints_path, self.config.game), self.config.game), global_step=step)

    def play(self):
        self.environment.new_game()
        x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.observe_display)
        self.init_history(x_tp)

        for _ in range(random.randint(0, self.config.no_op_max)):
            x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.play_display)
            self.observe(x_tp, r_t, 0, terminal_tp)

        print_with_time('Playing...')
        self.program_phase = 'playing'
        play_step = 1
        play_game = 1
        while True:
            a_t = self.predict(self.history.get(), self.config.no_exploration)
            x_tp, r_t, terminal_tp = self.environment.action(a_t, self.config.action_repeat, self.config.play_display)
            self.observe(x_tp, r_t, a_t, terminal_tp)

            if self.environment.game_over():
                self.game_rewards = np.append(self.game_rewards, self.environment.reward_game)
                self.game_scores = np.append(self.game_scores, self.environment.score_game)

                print('game: {}\t\tscores: {}\t\t rewards: {}'.format(play_game, self.environment.score_game,  self.environment.reward_game))

                play_game += 1
                self.environment.new_game()
                x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.observe_display)
                self.init_history(x_tp)

                for _ in range(random.randint(0, self.config.no_op_max)):
                    x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.play_display)
                    self.observe(x_tp, r_t, 0, terminal_tp)

            if play_step % self.config.stats_print_frequency == 0:
                self.print_statistics(play_step, self.config.stats_print_frequency)
                self.init_statistics()

            play_step += 1

    def train(self):
        self.environment.new_game()
        x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.observe_display)
        self.init_history(x_tp)

        for _ in range(random.randint(0, self.config.no_op_max)):
            x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.observe_display)
            self.observe(x_tp, r_t, 0, terminal_tp)

        print_with_time('Observing...')
        self.program_phase = 'observe'
        for observe_step in range(1, self.config.replay_start_size + 1):
            # a_t = self.predict(self.history.get(), self.config.initial_exploration)
            a_t = self.predict(self.history.get())
            x_tp, r_t, terminal_tp = self.environment.action(a_t, self.config.action_repeat, self.config.observe_display)
            self.observe(x_tp, r_t, a_t, terminal_tp)

            if self.environment.game_over():
                self.game_rewards = np.append(self.game_rewards, self.environment.reward_game)
                self.game_scores = np.append(self.game_scores, self.environment.score_game)

                self.environment.new_game()
                x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.observe_display)
                self.init_history(x_tp)

                for _ in range(random.randint(0, self.config.no_op_max)):
                    x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.observe_display)
                    self.observe(x_tp, r_t, 0, terminal_tp)

        self.print_statistics(observe_step, self.config.replay_start_size)
        self.init_statistics()

        print_with_time('Update target Q networks...')
        self.update_target_network()
        print_with_time('Training...')
        self.program_phase = 'explore'
        while self.step <= self.config.final_training:
            a_t = self.predict(self.history.get())
            x_tp, r_t, terminal_tp = self.environment.action(a_t, self.config.action_repeat, self.config.train_display)
            self.observe(x_tp, r_t, a_t, terminal_tp)

            if self.environment.game_over():
                self.game_rewards = np.append(self.game_rewards, self.environment.reward_game)
                self.game_scores = np.append(self.game_scores, self.environment.score_game)

                self.environment.new_game()
                self.episode += 1
                x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.observe_display)
                self.init_history(x_tp)

                for _ in range(random.randint(0, self.config.no_op_max)):
                    x_tp, r_t, terminal_tp = self.environment.action(0, self.config.action_repeat, self.config.train_display)
                    self.observe(x_tp, r_t, 0, terminal_tp)

            if self.step % self.config.stats_print_frequency == 0:
                if self.program_phase != 'train' and self.step >= self.config.final_exploration_frame:
                    self.program_phase = 'train'
                self.print_statistics(self.step, self.config.stats_print_frequency)
                self.init_statistics()

            if self.step % self.config.summary_frequency == 0:
                if self.config.create_summaries and self.predicted_Q_max.size > 0 and self.game_rewards.size > 0:
                    [summary] = self.sess.run([self.summary], feed_dict={
                        self.s_t: [self.history.get()],
                        self.target_s_t: [self.history.get()],
                        self.scores_summary: self.game_scores,
                        self.rewards_summary: self.game_rewards,
                        self.Q_max_summary: self.predicted_Q_max,
                        self.actions_summary: self.predicted_actions
                    })

                    self.summary_writer.add_summary(summary, self.step)
                    self.init_summary()

            if self.step % self.config.target_network_update_frequency == 0:
                print_with_time('Update target Q networks...')
                self.update_target_network()

            if self.config.create_checkpoints and self.step % self.config.model_save_frequency == 0:
                self.update_ckpt_step(self.step)
                self.save_model(self.step)

            if self.step % self.config.update_frequency == 0:
                self.minibatch(self.config.double_dqn)

            self.step += 1
