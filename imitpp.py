#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Imitation Learning for Point Process

A Python utility for generating marked spatial-temporal points by imitating expert sequences.

References:
- https://arxiv.org/abs/1811.05016

Dependencies:
- Python 3.6.7
- tensorflow==1.5.0
"""

import sys
import arrow
import utils
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats

class CustomizedStochasticLSTM(object):
    """
    Customized Stochastic LSTM Network

    A LSTM Network with customized stochastic output neurons, which used to generate time, location and marks accordingly.
    """

    def __init__(self, step_size, lstm_hidden_size, loc_hidden_size, mak_hidden_size, m_dim):
        """
        Params:
        - step_size:        the steps (length) of the LSTM network
        - lstm_hidden_size: size of hidden state of the LSTM
        - loc_hidden_size:  size of hidden feature of location
        - mak_hidden_size:  size of hidden feature of mark
        - m_dim:            number of categories of marks
        """
        INIT_PARAM_RATIO = 1e-5
        
        # data dimension
        self.t_dim = 1     # by default
        self.m_dim = m_dim # number of categories for the marks

        # model hyper-parameters
        self.step_size        = step_size        # step size of LSTM
        self.lstm_hidden_size = lstm_hidden_size # size of LSTM hidden feature
        self.loc_hidden_size  = loc_hidden_size  # size of location hidden feature
        self.loc_param_size   = 5                # by default
        self.mak_hidden_size  = mak_hidden_size  # size of mark hidden feature

        # define learning weights
        # - time weights
        self.Wt  = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.t_dim]))
        self.bt  = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.t_dim]))
        # - location weights
        self.Wl0 = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.loc_hidden_size]))
        self.bl0 = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.loc_hidden_size]))
        self.Wl1 = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.loc_hidden_size, self.loc_param_size]))
        self.bl1 = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.loc_param_size]))
        # - mark weights
        self.Wm0 = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.mak_hidden_size]))
        self.bm0 = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.mak_hidden_size]))
        self.Wm1 = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.mak_hidden_size, self.m_dim]))
        self.bm1 = tf.Variable(INIT_PARAM_RATIO * tf.random_normal([self.m_dim]))

    def initialize_network(self, batch_size):
        """Create a new network for training purpose, where the LSTM is at the zero state"""
        # create a basic LSTM cell
        tf_lstm_cell    = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size)
        # defining initial basic LSTM hidden state [2, batch_size, lstm_hidden_size]
        # - lstm_state.h: hidden state [batch_size, lstm_hidden_size]
        # - lstm_state.c: cell state   [batch_size, lstm_hidden_size]
        init_lstm_state = tf_lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # construct customized LSTM network
        self.seq_t, self.seq_l, self.seq_m, self.seq_loglik, self.final_state = self._recurrent_structure(
            batch_size, tf_lstm_cell, init_lstm_state)
        # # TODO: Debug
        # self.test = self.seq_m

    def _recurrent_structure(self, 
            batch_size, 
            tf_lstm_cell,     # tensorflow LSTM cell object, e.g. 'tf.nn.rnn_cell.BasicLSTMCell'
            init_lstm_state): # initial LSTM state tensor
        """Recurrent structure with customized LSTM cells."""
        # defining initial data point
        # - init_t: initial time     [batch_size, t_dim] 
        init_t = tf.zeros([batch_size, self.t_dim], dtype=tf.float32)
        # concatenate each customized LSTM cell by loop
        seq_t      = [] # generated sequence initialization
        seq_l      = []
        seq_m      = []
        seq_loglik = []
        last_t, last_lstm_state = init_t, init_lstm_state # loop initialization
        for _ in range(self.step_size):
            t, l, m, loglik, state = self._customized_lstm_cell(batch_size, tf_lstm_cell, last_lstm_state, last_t)
            seq_t.append(t)           # record generated time 
            seq_l.append(l)           # record generated location
            seq_m.append(m)           # record generated mark 
            seq_loglik.append(loglik) # record log likelihood  
            last_t          = t       # reset last_t
            last_lstm_state = state   # reset last_lstm_state
        seq_t      = tf.stack(seq_t, axis=1)      # [batch_size, step_size, t_dim]
        seq_l      = tf.stack(seq_l, axis=1)      # [batch_size, step_size, 2]
        seq_m      = tf.stack(seq_m, axis=1)      # [batch_size, step_size, m_dim]
        seq_loglik = tf.stack(seq_loglik, axis=1) # [batch_size, step_size, 1]
        return seq_t, seq_l, seq_m, seq_loglik, state

    def _customized_lstm_cell(self, batch_size, 
            tf_lstm_cell, # tensorflow LSTM cell object, e.g. 'tf.nn.rnn_cell.BasicLSTMCell'
            last_state,   # last state as input of this LSTM cell
            last_t):      # last_t + delta_t as input of this LSTM cell
        """
        Customized Stochastic LSTM Cell

        The customized LSTM cell takes current (time 't', location 'l', mark 'm') and the hidden state of last moment
        as input, return the ('next_t', 'next_l', 'next_m') as well as the hidden state for the next moment. The time,
        location and mark will be sampled based upon last hidden state.

        The reason avoid using tensorflow builtin rnn structure is that, besides last hidden state, the other feedback 
        to next moment is a customized stochastic variable which depends on the last moment's rnn output. 
        """
        # stochastic neurons for generating time, location and mark
        delta_t, loglik_t = self._dt(batch_size, last_state.h) # [batch_size, t_dim], [batch_size, 1] 
        next_l,  loglik_l = self._l(batch_size, last_state.h)  # [batch_size, 2],     [batch_size, 1] 
        next_m,  loglik_m = self._m(batch_size, last_state.h)  # [batch_size, m_dim], [batch_size, 1]  
        next_t = last_t + delta_t                              # [batch_size, t_dim]
        # log likelihood
        loglik = loglik_t # + loglik_l # + loglik_m    # TODO: Add mark to input x
        # input of LSTM
        x      = tf.concat([next_t], axis=1) # TODO: Add mark to input x
        # one step rnn structure
        # - x is a tensor that contains a single step of data points with shape [batch_size, t_dim + l_dim + m_dim]
        # - state is a tensor of hidden state with shape [2, batch_size, state_size]
        _, next_state = tf.nn.static_rnn(tf_lstm_cell, [x], initial_state=last_state, dtype=tf.float32)
        return next_t, next_l, next_m, loglik, next_state

    def _dt(self, batch_size, hidden_state):
        """Sampling time interval given hidden state of LSTM"""
        theta_h = tf.nn.elu(tf.matmul(hidden_state, self.Wt) + self.bt) + 1                         # [batch_size, t_dim=1]
        # reparameterization trick for sampling action from exponential distribution
        delta_t = - tf.log(tf.random_uniform([batch_size, self.t_dim], dtype=tf.float32)) / theta_h # [batch_size, t_dim=1]
        # log likelihood
        loglik  = - tf.multiply(theta_h, delta_t) + tf.log(theta_h)                                 # [batch_size, 1]
        return delta_t, loglik

    def _l(self, batch_size, hidden_state):
        """Sampling location shifts given hidden state of LSTM"""
        dense_feature = tf.nn.relu(tf.matmul(hidden_state, self.Wl0)) + self.bl0  # [batch_size, loc_hidden_size]
        dense_feature = tf.matmul(dense_feature, self.Wl1) + self.bl1             # [batch_size, loc_param_size]
        # 5 params that determine the distribution of location shifts with shape [batch_size]
        mu0 = tf.reshape(dense_feature[:, 0], [batch_size, 1]) 
        mu1 = tf.reshape(dense_feature[:, 1], [batch_size, 1])
        # construct positive definite and symmetrical matrix as covariance matrix
        A11 = tf.expand_dims(tf.reshape(dense_feature[:, 2], [batch_size, 1]), -1) # [batch_size, 1, 1]
        A22 = tf.expand_dims(tf.reshape(dense_feature[:, 3], [batch_size, 1]), -1) # [batch_size, 1, 1]
        A21 = tf.expand_dims(tf.reshape(dense_feature[:, 4], [batch_size, 1]), -1) # [batch_size, 1, 1]
        A12 = tf.zeros([batch_size, 1, 1])                                         # [batch_size, 1, 1]
        A1  = tf.concat([A11, A12], axis=2) # [batch_size, 1, 2]
        A2  = tf.concat([A21, A22], axis=2) # [batch_size, 1, 2]
        A   = tf.concat([A1, A2], axis=1)   # [batch_size, 2, 2]
        # sigma = A * A^T with shape [batch_size, 2, 2]
        sigma   = tf.scan(lambda a, x: tf.matmul(x, tf.transpose(x)), A) # [batch_size, 2, 2]
        sigma11 = tf.expand_dims(sigma[:, 0, 0], -1)                     # [batch_size, 1]
        sigma22 = tf.expand_dims(sigma[:, 1, 1], -1)                     # [batch_size, 1]
        sigma12 = tf.expand_dims(sigma[:, 0, 1], -1)                     # [batch_size, 1]
        # random variable for generating locaiton
        rv0 = tf.random_normal([batch_size, 1])
        rv1 = tf.random_normal([batch_size, 1])
        # location x and y
        x = mu0 + tf.multiply(sigma11, rv0) + tf.multiply(sigma12, rv1) # [batch_size, 1]
        y = mu1 + tf.multiply(sigma12, rv0) + tf.multiply(sigma22, rv1) # [batch_size, 1]
        l = tf.concat([x, y], axis=1)                                   # [batch_size, 2]
        # log likelihood
        sigma1 = tf.sqrt(tf.square(sigma11) + tf.square(sigma12))
        sigma2 = tf.sqrt(tf.square(sigma12) + tf.square(sigma22))
        v12 = tf.multiply(sigma11, sigma12) + tf.multiply(sigma12, sigma22)
        rho = v12 / tf.multiply(sigma1, sigma2)
        z   = tf.square(x - mu0) / tf.square(sigma1) \
            - 2 * tf.multiply(rho, tf.multiply(x - mu0, y - mu1)) / tf.multiply(sigma1, sigma2) \
            + tf.square(y - mu1) / tf.square(sigma2)
        loglik = - z / 2 / (1 - tf.square(rho)) \
                 - tf.log(2 * np.pi * tf.multiply(tf.multiply(sigma1, sigma2), tf.sqrt(1 - tf.square(rho))))
        return l, loglik
    
    def _m(self, batch_size, hidden_state):
        """Sampling mark given hidden state of LSTM"""
        dense_feature = tf.nn.relu(tf.matmul(hidden_state, self.Wm0)) + self.bm0      # [batch_size, location_para_dim]
        dense_feature = tf.nn.elu(tf.matmul(dense_feature, self.Wm1) + self.bm1) + 1  # [batch_size, dim_m] dense_feature is positive
        # sample from multinomial distribution (use Gumbel trick to sample the labels)
        eps        = 1e-13
        rv_uniform = tf.random_uniform([batch_size, self.m_dim])
        rv_Gumbel  = -tf.log(-tf.log(rv_uniform + eps) + eps)
        label      = tf.argmax(dense_feature + rv_Gumbel, axis=1) # label: [batch]
        m          = tf.one_hot(indices=label, depth=self.m_dim)  # [batch_size, m_dim]
        # log likelihood
        prob       = tf.nn.softmax(dense_feature)
        loglik     = tf.log(tf.reduce_sum(m * prob, 1) + 1e-13)
        return m, loglik



class PointProcessGenerator(object):
    """
    Point Process Generator
    """

    def __init__(self, T, seq_len, lstm_hidden_size, loc_hidden_size, mak_hidden_size, m_dim):
        """
        Params:
        - T:                the maximum time of the sequences
        - seq_len:          the length of the sequences
        - lstm_hidden_size: size of hidden state of the LSTM
        - loc_hidden_size:  size of hidden feature of location
        - mak_hidden_size:  size of hidden feature of mark
        - m_dim:            number of categories of marks
        """
        # model hyper-parameters
        self.T       = T                # maximum time
        self.t_dim   = 1                # by default
        self.l_dim   = 2                # by default
        self.m_dim   = m_dim            # number of categories of marks
        self.seq_len = seq_len          # length of each generated sequences
        # LSTM generator
        self.cslstm  = CustomizedStochasticLSTM(
            step_size=seq_len, lstm_hidden_size=lstm_hidden_size, 
            loc_hidden_size=loc_hidden_size, mak_hidden_size=mak_hidden_size, m_dim=m_dim)
    
    def _initialize_policy_network(self, batch_size, starter_learning_rate=0.01, decay_rate=0.99, decay_step=100):
        """
        Construct Policy Network
        
        Policy should be flexible and expressive enough to capture the potential complex point process patterns of data.
        Therefore, a customized recurrent neural network (RNN) with stochastic neurons is adopted, where hidden state is 
        computed by hidden state of last moment and stochastically generated action. i.e.
          a_{i+1} is sampling from pi(a|h_{i})
          h_{i+1} = rnn_cell(h_{i}, a_{i+1})
        """
        # input tensors: expert sequences (time, location, marks)
        self.input_seq_t = tf.placeholder(tf.float32, [batch_size, None, self.t_dim])
        self.input_seq_l = tf.placeholder(tf.float32, [batch_size, None, self.l_dim])
        self.input_seq_m = tf.placeholder(tf.float32, [batch_size, None, self.m_dim])

        # construct customized stochastic LSTM network
        self.cslstm.initialize_network(batch_size)
        # generated tensors: learner sequences (time, location, marks)
        learner_seq_t, learner_seq_l, learner_seq_m = self.cslstm.seq_t, self.cslstm.seq_l, self.cslstm.seq_m
        # log likelihood
        learner_seq_loglik = self.cslstm.seq_loglik

        # getting training time window (t_0 = 0, T = self.T by default)
        t0, T = self._training_time_window(learner_seq_t)

        # concatenate batches in the sequences
        expert_seq_t,  expert_seq_l,  expert_seq_m  = \
            self.__concatenate_batch(self.input_seq_t), \
            self.__concatenate_batch(self.input_seq_l), \
            self.__concatenate_batch(self.input_seq_m)
        learner_seq_t, learner_seq_l, learner_seq_m, learner_seq_loglik = \
            self.__concatenate_batch(learner_seq_t), \
            self.__concatenate_batch(learner_seq_l), \
            self.__concatenate_batch(learner_seq_m), \
            self.__concatenate_batch(learner_seq_loglik)
        
        # calculate average rewards
        reward = self._reward(batch_size, t0, T,\
                              expert_seq_t,  expert_seq_l,  expert_seq_m, \
                              learner_seq_t, learner_seq_l, learner_seq_m) # [batch_size*seq_len, 1]

        # cost and optimizer
        self.cost      = tf.reduce_sum(tf.multiply(reward, learner_seq_loglik), axis=0) / batch_size
        global_step    = tf.Variable(0, trainable=False)
        learning_rate  = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.6, beta2=0.9).minimize(self.cost, global_step=global_step)

    def _training_time_window(self, learner_seq_t):
        """
        Time window for the purpose of training. The model only fits a specific segment of the expert sequence
        indicated by 'training_time_window'. This function will return the start time (t_0) and end time (T) of 
        the segment.

        Policy 1:
        t_0 = 0; T = mean(max(learner_seq_t, axis=0))
        """
        # remove invalid time
        mask_t        = self.__get_mask_truncate_by_T(learner_seq_t, self.T, 0) # [batch_size, seq_len, 1]
        learner_seq_t = tf.multiply(learner_seq_t, mask_t)                      # [batch_size, seq_len, 1]
        # policy 1
        t_0 = 0
        T   = tf.reduce_mean(tf.reduce_max(learner_seq_t, axis=0))
        return t_0, T

    def _reward(self, batch_size, t0, T, 
            expert_seq_t, expert_seq_l, expert_seq_m,    # expert sequences
            learner_seq_t, learner_seq_l, learner_seq_m, # learner sequences
            kernel_bandwidth=1): 
        """reward function"""
        # get mask for concatenated expert and learner sequences
        expert_seq_mask  = self.__get_mask_truncate_by_T(expert_seq_t, T, t0)  # batch_size*seq_len
        learner_seq_mask = self.__get_mask_truncate_by_T(learner_seq_t, T, t0) # batch_size*seq_len
        # calculate mask for kernel matrix
        learner_learner_kernel_mask = tf.matmul(learner_seq_mask, tf.transpose(learner_seq_mask))
        expert_learner_kernel_mask  = tf.matmul(expert_seq_mask, tf.transpose(learner_seq_mask))
        # concatenate each data dimension for both expert sequence and learner sequence
        # TODO: Add mark to the sequences
        # expert_seq  = tf.concat([expert_seq_t, expert_seq_l], axis=1)   # [batch_size*seq_len, t_dim+l_dim+m_dim]
        # learner_seq = tf.concat([learner_seq_t, learner_seq_l], axis=1) # [batch_size*seq_len, t_dim+l_dim+m_dim]
        expert_seq  = tf.concat([expert_seq_t], axis=1)                          # [batch_size*seq_len, t_dim]
        learner_seq = tf.concat([learner_seq_t], axis=1)                         # [batch_size*seq_len, t_dim]
        # calculate upper-half kernel matrix
        learner_learner_kernel, expert_learner_kernel = self.__kernel_matrix(
            learner_seq, expert_seq, kernel_bandwidth)                           # 2 * [batch_size*seq_len, batch_size*seq_len]
        learner_learner_kernel = tf.multiply(learner_learner_kernel, learner_learner_kernel_mask)
        expert_learner_kernel  = tf.multiply(expert_learner_kernel, expert_learner_kernel_mask)
        # calculate reward for each of data point in learner sequence
        emp_ll_mean = tf.reduce_sum(learner_learner_kernel, axis=0) / batch_size # batch_size*seq_len
        emp_el_mean = tf.reduce_sum(expert_learner_kernel, axis=0) / batch_size  # batch_size*seq_len
        return tf.expand_dims(emp_ll_mean - emp_el_mean, -1)                     # [batch_size*seq_len, 1]

    @staticmethod
    def __get_mask_truncate_by_T(seq_t, T, t_0=0):
        """Masking time, location and mark sequences for the entries before the maximum time T."""
        # squeeze since each time sequence has shape [batch_size*seq_len, 1] or [batch_size, seq_len, 1]
        array_t = tf.squeeze(seq_t)                  # batch_size*seq_len or [batch_size, seq_len]
        # get basic mask where 0 if t > T else 1
        mask_t  = tf.expand_dims(tf.multiply(
            tf.cast(array_t < T, tf.float32),
            tf.cast(array_t > t_0, tf.float32)), -1) # [batch_size*seq_len, 1] or [batch_size, seq_len, 1]
        return mask_t
    
    @staticmethod
    def __concatenate_batch(seqs):
        """Concatenate each batch of the sequences into a single sequence."""
        array_seq = tf.unstack(seqs, axis=0)     # [batch_size, seq_len, data_dim]
        seq       = tf.concat(array_seq, axis=0) # [batch_size*seq_len, data_dim]
        return seq
 
    @staticmethod
    def __kernel_matrix(learner_seq, expert_seq, kernel_bandwidth):
        """
        Construct kernel matrix based on learn sequence and expert sequence, each entry of the matrix 
        is the distance between two data points in learner_seq or expert_seq. return two matrix, left_mat 
        is the distances between learn sequence and learn sequence, right_mat is the distances between 
        learn sequence and expert sequence.
        """
        # calculate l2 distances
        learner_learner_mat = utils.l2_norm(learner_seq, learner_seq) # [batch_size*seq_len, batch_size*seq_len]
        expert_learner_mat  = utils.l2_norm(expert_seq, learner_seq)  # [batch_size*seq_len, batch_size*seq_len]
        # exponential kernel
        learner_learner_mat = tf.exp(-learner_learner_mat / kernel_bandwidth)
        expert_learner_mat  = tf.exp(-expert_learner_mat / kernel_bandwidth)
        return learner_learner_mat, expert_learner_mat

    def train(self, sess, batch_size, 
            epoches,               # number of epoches (how many times is the entire dataset going to be trained)
            expert_seq_t,          # [n, seq_len, 1]
            expert_seq_l,          # [n, seq_len, 2]
            expert_seq_m,          # [n, seq_len, m_dim]
            train_test_ratio = 9., # n_train / n_test
            trainplot=True,        # plot the change of intensity over epoches
            pretrained=False):
        """Train the point process generator given expert sequences."""
        # check the consistency of the shape of the expert sequences
        assert expert_seq_t.shape[:-1] == expert_seq_l.shape[:-1] == expert_seq_m.shape[:-1], \
            "inconsistant 'number of sequences' or 'sequence length' of input expert sequences"

        # initialization
        if not pretrained:
            # initialize policy network
            self._initialize_policy_network(batch_size)
            # initialize network parameters
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

        # data configurations
        # - number of expert sequences
        n_data  = expert_seq_t.shape[0]
        n_train = int(n_data * train_test_ratio / (train_test_ratio + 1.))
        n_test  = int(n_data * 1. / (train_test_ratio + 1.))
        # - number of batches
        n_batches = int(n_train / batch_size)
        # - check if test data size is large enough (> batch_size)
        assert n_test >= batch_size, "test data size %d is less than batch size %d." % (n_test, batch_size)
        
        if trainplot:
            ppim = utils.PointProcessIntensityMeter(self.T, batch_size)

        # training over epoches
        for epoch in range(epoches):
            # shuffle indices of the training samples
            shuffled_ids = np.arange(n_data)
            np.random.shuffle(shuffled_ids)
            shuffled_train_ids = shuffled_ids[:n_train]
            shuffled_test_ids  = shuffled_ids[-n_test:]

            # training over batches
            avg_train_cost = []
            avg_test_cost  = []
            for b in range(n_batches):
                idx             = np.arange(batch_size * b, batch_size * (b + 1))
                # training and testing indices selected in current batch
                batch_train_ids = shuffled_train_ids[idx]
                batch_test_ids  = shuffled_test_ids[:batch_size]
                # training and testing batch data
                batch_train_expert_t = expert_seq_t[batch_train_ids, :, :]
                batch_train_expert_l = expert_seq_l[batch_train_ids, :, :]
                batch_train_expert_m = expert_seq_m[batch_train_ids, :, :]
                batch_test_expert_t  = expert_seq_t[batch_test_ids, :, :]
                batch_test_expert_l  = expert_seq_l[batch_test_ids, :, :]
                batch_test_expert_m  = expert_seq_m[batch_test_ids, :, :]
                # optimization procedure
                sess.run(self.optimizer, feed_dict={
                    self.input_seq_t: batch_train_expert_t,
                    self.input_seq_l: batch_train_expert_l,
                    self.input_seq_m: batch_train_expert_m})
                # cost for train batch and test batch
                train_cost = sess.run(self.cost, feed_dict={
                    self.input_seq_t: batch_train_expert_t,
                    self.input_seq_l: batch_train_expert_l,
                    self.input_seq_m: batch_train_expert_m})
                test_cost  = sess.run(self.cost, feed_dict={
                    self.input_seq_t: batch_test_expert_t,
                    self.input_seq_l: batch_test_expert_l,
                    self.input_seq_m: batch_test_expert_m})
                # record cost for each batch
                avg_train_cost.append(train_cost)
                avg_test_cost.append(test_cost)

            if trainplot:
                # update intensity plot
                learner_seq_t, learner_seq_l = sess.run(
                    [self.cslstm.seq_t, self.cslstm.seq_l], 
                    feed_dict={
                        self.input_seq_t: batch_test_expert_t,
                        self.input_seq_l: batch_test_expert_l,
                        self.input_seq_m: batch_test_expert_m})
                ppim.update_time_intensity(batch_train_expert_t, learner_seq_t)
                ppim.update_location_intensity(batch_train_expert_l, learner_seq_l)

            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            avg_test_cost  = np.mean(avg_test_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % (arrow.now(), epoch, n_batches, batch_size), file=sys.stderr)
            print('[%s] Training cost:\t%f' % (arrow.now(), avg_train_cost), file=sys.stderr)
            print('[%s] Testing cost:\t%f' % (arrow.now(), avg_test_cost), file=sys.stderr)