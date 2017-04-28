import time
import tensorflow as tf
import numpy as np

def print_with_time(text):
    print('[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), text))

def huber_loss(error):
    with tf.variable_scope('huber_loss'):
        return tf.where(tf.abs(error) < 1.0, 0.5 * tf.square(error), tf.abs(error) - 0.5)

def conv_relu(input, kernel_shape, stride, channels_in, channels_out, padding, name='conv_relu'):
    with tf.variable_scope(name):
        strides = [1, stride, stride, 1]

        w = tf.get_variable('W', shape=kernel_shape + [channels_in, channels_out], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('B', shape=[channels_out], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input, w, strides=strides, padding=padding)
        act = tf.nn.relu(conv + b)

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)

        return act, w, b

def fc_linear(input, input_size, output_size, name='fc_linear'):
    with tf.variable_scope(name):
        w = tf.get_variable('W', shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('B', shape=[output_size], initializer=tf.constant_initializer(0.0))
        linear = tf.matmul(input, w) + b

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('linears', linear)

        return linear, w, b

def fc_relu(input, input_size, output_size, name='fc_relu'):
    with tf.variable_scope(name):
        w = tf.get_variable('W', shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('B', shape=[output_size], initializer=tf.constant_initializer(0.0))
        act = tf.nn.relu(tf.matmul(input, w) + b)

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)

        return act, w, b

def variable_summaries(var, name='summaries'):
    with tf.variable_scope(name):
        with tf.variable_scope('stats'):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
