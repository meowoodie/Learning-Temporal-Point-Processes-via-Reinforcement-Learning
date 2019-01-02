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
from scipy import stats
import matplotlib.pyplot as plt

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
        self.Wt  = tf.Variable(tf.random_normal([self.lstm_hidden_size, self.t_dim]))
        self.bt  = tf.Variable(tf.random_normal([self.t_dim]))
        # - location weights
        self.Wl0 = tf.Variable(tf.random_normal([self.lstm_hidden_size, self.loc_hidden_size]))
        self.bl0 = tf.Variable(tf.random_normal([self.loc_hidden_size]))
        self.Wl1 = tf.Variable(tf.random_normal([self.loc_hidden_size, self.loc_param_size]))
        self.bl1 = tf.Variable(tf.random_normal([self.loc_param_size]))
        # - mark weights
        self.Wm0 = tf.Variable(tf.random_normal([self.lstm_hidden_size, self.mak_hidden_size]))
        self.bm0 = tf.Variable(tf.random_normal([self.mak_hidden_size]))
        self.Wm1 = tf.Variable(tf.random_normal([self.mak_hidden_size, self.m_dim]))
        self.bm1 = tf.Variable(tf.random_normal([self.m_dim]))

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
        seq_t      = tf.stack(seq_t, axis=1)
        seq_l      = tf.stack(seq_l, axis=1) 
        seq_m      = tf.stack(seq_m, axis=1)
        seq_loglik = tf.stack(seq_loglik, axis=1)
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
        loglik = loglik_t + loglik_l # + loglik_m # TODO: Add mark to input x
        # input of LSTM
        x      = tf.concat([next_t, next_l], axis=1) # TODO: Add mark to input x
        # one step rnn structure
        # - x is a tensor that contains a single step of data points with shape [batch_size, t_dim + l_dim + m_dim]
        # - state is a tensor of hidden state with shape [2, batch_size, state_size]
        _, next_state = tf.nn.static_rnn(tf_lstm_cell, [x], initial_state=last_state, dtype=tf.float32)
        return next_t, next_l, next_m, loglik, next_state

    def _dt(self, batch_size, hidden_state):
        """Sampling time interval given hidden state of LSTM"""
        theta_h = tf.nn.elu(tf.matmul(hidden_state, self.Wt) + self.bt) + 1                         # [batch_size, t_dim]
        # reparameterization trick for sampling action from exponential distribution
        delta_t = - tf.log(tf.random_uniform([batch_size, self.t_dim], dtype=tf.float32)) / theta_h # [batch_size, t_dim]
        # log likelihood
        loglik  = - tf.multiply(theta_h, delta_t) + tf.log(theta_h)                                 # [batch_size, 1]
        return delta_t, loglik 

    def _l(self, batch_size, hidden_state):
        """Sampling location shifts given hidden state of LSTM"""
        dense_feature = tf.nn.relu(tf.matmul(hidden_state, self.Wl0)) + self.bl0  # [batch_size, loc_hidden_size]
        dense_feature = tf.matmul(dense_feature, self.Wl1) + self.bl1             # [batch_size, loc_param_size]
        # 5 params that determine the distribution of location shifts with shape [batch_size]
        mu0     = tf.reshape(dense_feature[:, 0], [batch_size, 1]) 
        mu1     = tf.reshape(dense_feature[:, 1], [batch_size, 1])
        sigma11 = tf.reshape(tf.exp(dense_feature[:, 2]), [batch_size, 1])
        sigma22 = tf.reshape(tf.exp(dense_feature[:, 3]), [batch_size, 1])
        sigma12 = tf.reshape(dense_feature[:, 4], [batch_size, 1])
        # random variable for generating locaiton
        rv0 = tf.random_normal([batch_size, 1])
        rv1 = tf.random_normal([batch_size, 1])
        # location x and y
        x = mu0 + tf.multiply(sigma11, rv0) + tf.multiply(sigma12, rv1)
        y = mu1 + tf.multiply(sigma12, rv0) + tf.multiply(sigma22, rv1)
        l = tf.concat([x, y], axis=1) # [batch_size, 2]
        # log likelihood
        sigma1 = tf.sqrt(tf.square(sigma11) + tf.square(sigma12))
        sigma2 = tf.sqrt(tf.square(sigma12) + tf.square(sigma22))
        v12 = tf.multiply(sigma11, sigma12) + tf.multiply(sigma12, sigma22)
        rho = v12 / tf.multiply(sigma1, sigma2)
        z   = tf.square(x - mu0) / tf.square(sigma1) \
            - 2 * tf.multiply(rho, tf.multiply(x - mu0, y - mu1)) / tf.multiply(sigma1, sigma2) \
            + tf.square(y - mu1) / tf.square(sigma2)
        loglik = -z / 2 / (1 - tf.square(rho)) - tf.log(
            2 * np.pi * tf.multiply(tf.multiply(sigma1, sigma2), tf.sqrt(1 - tf.square(rho))))
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
        
        # sequences preprocessing
        # - truncate sequences after time T
        expert_seq_t, expert_seq_l, expert_seq_m, = \
            self.__truncate_by_T(self.input_seq_t, T=self.T, seq_t=self.input_seq_t), \
            self.__truncate_by_T(self.input_seq_l, T=self.T, seq_t=self.input_seq_t), \
            self.__truncate_by_T(self.input_seq_m, T=self.T, seq_t=self.input_seq_t)
        learner_seq_t, learner_seq_l, learner_seq_m, learner_seq_loglik = \
            self.__truncate_by_T(learner_seq_t, T=self.T, seq_t=self.input_seq_t), \
            self.__truncate_by_T(learner_seq_l, T=self.T, seq_t=self.input_seq_t), \
            self.__truncate_by_T(learner_seq_m, T=self.T, seq_t=self.input_seq_t), \
            self.__truncate_by_T(learner_seq_loglik, T=self.T, seq_t=self.input_seq_t)
        # - concatenate batches in the sequences
        expert_seq_t,  expert_seq_l,  expert_seq_m  = \
            self.__concatenate_batch(expert_seq_t), \
            self.__concatenate_batch(expert_seq_l), \
            self.__concatenate_batch(expert_seq_m)
        learner_seq_t, learner_seq_l, learner_seq_m, learner_seq_loglik = \
            self.__concatenate_batch(learner_seq_t), \
            self.__concatenate_batch(learner_seq_l), \
            self.__concatenate_batch(learner_seq_m), \
            self.__concatenate_batch(learner_seq_loglik)
        
        # calculate average rewards
        reward = self._reward(batch_size, \
                              expert_seq_t,  expert_seq_l,  expert_seq_m, \
                              learner_seq_t, learner_seq_l, learner_seq_m) # [x*seq_len, 1]

        # cost and optimizer
        self.cost      = tf.reduce_sum(tf.multiply(reward, learner_seq_loglik), axis=0) / batch_size
        global_step    = tf.Variable(0, trainable=False)
        learning_rate  = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.6, beta2=0.9).minimize(self.cost, global_step=global_step)
        # self.test = self.cost

    def _reward(self, batch_size,
            expert_seq_t, expert_seq_l, expert_seq_m,    # expert sequences
            learner_seq_t, learner_seq_l, learner_seq_m, # learner sequences
            kernel_bandwidth=1): 
        """reward function"""
        # concatenate each data dimension for both expert sequence and learner sequence
        # TODO: Add mark to the sequences
        expert_seq  = tf.concat([expert_seq_t, expert_seq_l], axis=1)    # [batch_size*seq_len, t_dim+l_dim+m_dim]
        learner_seq = tf.concat([learner_seq_t, learner_seq_l], axis=1) # [batch_size*seq_len, t_dim+l_dim+m_dim]
        # calculate upper-half kernel matrix
        learner_learner_kernel, learner_expert_kernel = self.__kernel_matrix(learner_seq, expert_seq, kernel_bandwidth) # [batch_size*seq_len, 2*batch_size*seq_len]
        # calculate reward for each of data point in learner sequence
        emp_expert_mean  = tf.reduce_sum(learner_learner_kernel, axis=1) / batch_size   # batch_size*seq_len
        emp_learner_mean = tf.reduce_sum(learner_expert_kernel, axis=1) / batch_size # batch_size*seq_len
        return tf.expand_dims(emp_expert_mean - emp_learner_mean, -1) # [batch_size*seq_len, 1]

    def train(self, sess, batch_size, 
            epoches,               # number of epoches (how many times is the entire dataset going to be trained)
            expert_seq_t,          # [n, seq_len, 1]
            expert_seq_l,          # [n, seq_len, 2]
            expert_seq_m,          # [n, seq_len, m_dim]
            train_test_ratio = 9., # n_train / n_test
            pretrained=False):
        """train"""
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
            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            avg_test_cost  = np.mean(avg_test_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % (arrow.now(), epoch, n_batches, batch_size), file=sys.stderr)
            print('[%s] Training cost:\t%f' % (arrow.now(), avg_train_cost), file=sys.stderr)
            print('[%s] Testing cost:\t%f' % (arrow.now(), avg_test_cost), file=sys.stderr)
        
    @staticmethod
    def __truncate_by_T(trunc_seq, T, seq_t):
        """masking time, location and mark sequences for the entries before the maximum time T."""
        # data dimension of seq
        d = tf.shape(trunc_seq)[-1]
        # squeeze since each time entry is a list with a single element. 
        array_t = tf.squeeze(seq_t, squeeze_dims=[2]) # [batch_size, seq_len]
        # get basic mask where 0 if t > T else 1
        mask_t  = tf.expand_dims(tf.multiply(
            tf.cast(array_t < T, tf.float32),
            tf.cast(array_t > 0, tf.float32)), -1)    # [batch_size, seq_len, 1] 
        mask    = tf.tile(mask_t, [1, 1, d])          # [batch_size, seq_len, l_dim]
        # return masked sequences
        return tf.multiply(trunc_seq, mask)
    
    @staticmethod
    def __concatenate_batch(seqs):
        """concatenate each batch of the sequences into a single sequence."""
        array_seq = tf.unstack(seqs, axis=0)     # [batch_size, seq_len, data_dim]
        seq       = tf.concat(array_seq, axis=0) # [batch_size*seq_len, data_dim]
        return seq

    @staticmethod
    def __kernel_matrix(learner_seq, expert_seq, kernel_bandwidth):
        """
        construct kernel matrix based on learn sequence and expert sequence, each entry of the matrix 
        is the distance between two data points in learner_seq or expert_seq. return two matrix, left_mat 
        is the distances between learn sequence and learn sequence, right_mat is the distances between 
        learn sequence and expert sequence.
        """
        # helper function for getting nonzero 1D mask for a 2D sequence
        def nonzero_mask(seq):
            # data dimension
            d    = tf.shape(seq)[-1] 
            # 2D seq mask: 0 for zero, 1 for nonzero
            mask = tf.cast(seq > 0, tf.float32)                    # [seq_len, data_dim]
            mask = tf.expand_dims(tf.reduce_sum(mask, axis=1), -1) # [seq_len, 1]
            mask = tf.cast(mask > 0, tf.float32)                   # [seq_len, 1]
            return mask
        # calculate l2 distances
        learner_learner_mat = utils.l2_norm(learner_seq, learner_seq) # [batch_size*seq_len, batch_size*seq_len]
        learner_expert_mat  = utils.l2_norm(learner_seq, expert_seq)  # [batch_size*seq_len, batch_size*seq_len]
        # exponential kernel
        learner_learner_mat = tf.exp(-tf.square(learner_learner_mat) / kernel_bandwidth)
        learner_expert_mat  = tf.exp(-tf.square(learner_expert_mat) / kernel_bandwidth)
        # remove invalid entries
        learner_seq_mask = nonzero_mask(learner_seq)
        expert_seq_mask  = nonzero_mask(expert_seq)
        learner_learner_mat_mask = tf.matmul(learner_seq_mask, tf.transpose(expert_seq_mask))
        learner_expert_mat_mask  = tf.matmul(learner_seq_mask, tf.transpose(expert_seq_mask))
        learner_learner_mat = tf.multiply(learner_learner_mat, learner_learner_mat_mask)
        learner_expert_mat  = tf.multiply(learner_expert_mat, learner_expert_mat_mask)
        return learner_learner_mat, learner_expert_mat

    # def debug(self, sess, input_seq_t, input_seq_l, input_seq_m):
    #     return sess.run(self.test, feed_dict={
    #         self.input_seq_t: input_seq_t,
    #         self.input_seq_l: input_seq_l,
    #         self.input_seq_m: input_seq_m})

# if __name__ == "__main__":
#     expert_seq_t = [
#         [[ 2.2372603], 
#         [ 7.3469152],
#         [10.841639 ],
#         [11.278158 ],
#         [11.875915 ]],

#         [[ 4.601893 ],
#         [ 7.6262646],
#         [ 8.953828 ],
#         [11.48958  ],
#         [13.335195 ]],
        
#         [[ 2.2372603], 
#         [ 7.3469152],
#         [10.841639 ],
#         [11.278158 ],
#         [11.875915 ]],

#         [[ 4.601893 ],
#         [ 7.6262646],
#         [ 8.953828 ],
#         [11.48958  ],
#         [13.335195 ]],
        
#         [[ 2.2372603], 
#         [ 7.3469152],
#         [10.841639 ],
#         [11.278158 ],
#         [11.875915 ]],

#         [[ 4.601893 ],
#         [ 7.6262646],
#         [ 8.953828 ],
#         [11.48958  ],
#         [13.335195 ]]]

#     expert_seq_l = [
#         [[-9.7975151e+01,  2.7360342e+00],
#         [-5.2876039e+00, -6.6114247e-01],
#         [-7.1111503e+00, -8.5162185e-03],
#         [ 1.1980354e+01,  1.2619636e+00],
#         [ 3.5298267e+02,  5.9427624e+00]],

#         [[ 2.0113881e+03,  5.1461897e+00],
#         [ 4.9495239e+02,  5.6867313e+00],
#         [-5.8737720e+02,  1.8419909e+00],
#         [-1.5442281e+00, -1.3099791e+00],
#         [ 4.1468458e+00,  8.7737030e-01]],
        
#         [[-9.7975151e+01,  2.7360342e+00],
#         [-5.2876039e+00, -6.6114247e-01],
#         [-7.1111503e+00, -8.5162185e-03],
#         [ 1.1980354e+01,  1.2619636e+00],
#         [ 3.5298267e+02,  5.9427624e+00]],

#         [[ 2.0113881e+03,  5.1461897e+00],
#         [ 4.9495239e+02,  5.6867313e+00],
#         [-5.8737720e+02,  1.8419909e+00],
#         [-1.5442281e+00, -1.3099791e+00],
#         [ 4.1468458e+00,  8.7737030e-01]],
        
#         [[-9.7975151e+01,  2.7360342e+00],
#         [-5.2876039e+00, -6.6114247e-01],
#         [-7.1111503e+00, -8.5162185e-03],
#         [ 1.1980354e+01,  1.2619636e+00],
#         [ 3.5298267e+02,  5.9427624e+00]],

#         [[ 2.0113881e+03,  5.1461897e+00],
#         [ 4.9495239e+02,  5.6867313e+00],
#         [-5.8737720e+02,  1.8419909e+00],
#         [-1.5442281e+00, -1.3099791e+00],
#         [ 4.1468458e+00,  8.7737030e-01]]]
    
#     expert_seq_m = [
#         [[0., 0., 1.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.]],

#         [[1., 0., 0.],
#         [1., 0., 0.],
#         [1., 0., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.]],
        
#         [[0., 0., 1.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.]],

#         [[1., 0., 0.],
#         [1., 0., 0.],
#         [1., 0., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.]],
        
#         [[0., 0., 1.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.]],

#         [[1., 0., 0.],
#         [1., 0., 0.],
#         [1., 0., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.]]]

#     expert_seq_t, expert_seq_l, expert_seq_m = np.array(expert_seq_t), np.array(expert_seq_l), np.array(expert_seq_m)

#     tf.set_random_seed(1)
#     # Start training
#     with tf.Session() as sess:
#         step_size        = 5
#         lstm_hidden_size = 10
#         loc_hidden_size  = 10
#         mak_hidden_size  = 10
#         m_dim            = 3
#         batch_size       = 2
#         T                = 11.
#         epoches          = 10

#         ppg = PointProcessGenerator(
#             T=T, seq_len=step_size, 
#             lstm_hidden_size=lstm_hidden_size, loc_hidden_size=loc_hidden_size, mak_hidden_size=mak_hidden_size, 
#             m_dim=m_dim)

#         ppg.train(sess, 
#             batch_size, epoches, 
#             expert_seq_t, expert_seq_l, expert_seq_m,
#             train_test_ratio = 2.)

#         # ppg._initialize_policy_network(batch_size)
#         # # Initialize the variables (i.e. assign their default value)
#         # init = tf.global_variables_initializer()
#         # # Run the initializer
#         # sess.run(init)
#         # print(ppg.debug(sess, expert_seq_t, expert_seq_l, expert_seq_m))
        

    