# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

from __future__ import division
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from config import ModelConfig
import os
from utils import shape
import numpy as np


class Model:
    def __init__(self, n_rnn_layer=3, hidden_size=256):

        # Input, Output
        self.x_mixed_mag = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='x_mixed_mag')
        self.x_mixed_phase = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='x_mixed_phase')
        self.y_src1 = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='y_src1')
        self.y_src2 = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='y_src2')

        # Network
        self.hidden_size = hidden_size
        self.n_layer = n_rnn_layer
        self.net = tf.make_template('net', self._net) #wrap func _net with name 'net', reuse variable
        self()

    def __call__(self):
        return self.net()

    def _net(self):
        # RNN and dense layers
        rnn_layer = MultiRNNCell([GRUCell(self.hidden_size) for _ in range(self.n_layer)]) # create a RNN cell composed sequentially of a number of RNNCells
        output_rnn, rnn_state = tf.nn.dynamic_rnn(rnn_layer, self.x_mixed_mag, dtype=tf.float32) #Creates a recurrent neural network specified by RNNCell cell

        input_size = shape(self.x_mixed_mag)[2]

        phase_dense_1 = tf.layers.dense(inputs=self.x_mixed_phase, units=input_size, activation=tf.nn.relu, name='phase_dense_1')
        phase_dense_2 = tf.layers.dense(inputs=phase_dense_1, units=input_size, activation=tf.nn.relu, name='phase_dense_2')

        concatenate = tf.concat([output_rnn, phase_dense_2],2)
        concatenate_dense = tf.layers.dense(inputs=concatenate, units=input_size, activation=tf.nn.relu, name='concatenate_dense')
        y_hat_src1 = tf.layers.dense(inputs=concatenate_dense, units=input_size, activation=tf.nn.relu, name='y_hat_src1')
        y_hat_src2 = tf.layers.dense(inputs=concatenate_dense, units=input_size, activation=tf.nn.relu, name='y_hat_src2')

        # time-freq masking layer
        y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed_mag
        y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed_mag

        return y_tilde_src1, y_tilde_src2

    def loss(self):
        pred_y_src1, pred_y_src2 = self()
        # Mean squared error(MSE) 
        return (tf.reduce_mean(tf.square(self.y_src1 - pred_y_src1) + tf.square(self.y_src2 - pred_y_src2), name='loss'))/2

    @staticmethod
    # shape = (batch_size, n_freq, n_frames) => (batch_size, n_frames, n_freq)
    def spec_to_batch(src):
        num_wavs, freq, n_frames = src.shape

        # padding the sample frames to meet  n*n_frames
        pad_len = 0
        if n_frames % ModelConfig.SEQ_LEN > 0:
            pad_len = (ModelConfig.SEQ_LEN - (n_frames % ModelConfig.SEQ_LEN))
        pad_width = ((0, 0), (0, 0), (0, pad_len))
        padded_src = np.pad(src, pad_width=pad_width, mode='constant', constant_values=0)

        assert(padded_src.shape[-1] % ModelConfig.SEQ_LEN == 0) #reassure op

        batch = np.reshape(padded_src.transpose(0, 2, 1), (-1, ModelConfig.SEQ_LEN, freq))
        #print('shape of src:{} shape of padded_src:{} shape of batch:{}'.format(src.shape, padded_src.shape, batch.shape))
        return batch, padded_src

    @staticmethod
    def batch_to_spec(src, num_wav):
        # shape = (batch_size, n_frames, n_freq) => (batch_size, n_freq, n_frames)
        batch_size, seq_len, freq = src.shape
        src = np.reshape(src, (num_wav, -1, freq))
        src = src.transpose(0, 2, 1)
        return src

    @staticmethod
    def load_state(sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)