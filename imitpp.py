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
import random
import numpy as np
import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt

class CustomizedStochasticLSTM(object):
    """
    Customized Stochastic LSTM Network
    """

    def __init__(self, batch_size, step_size, lstm_hidden_size, loc_hidden_size, mak_hidden_size, m_dim):
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

        # create a basic LSTM cell
        tf_lstm_cell    = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size)
        # defining initial basic LSTM hidden state [2, batch_size, lstm_hidden_size]
        # - lstm_state.h: hidden state [batch_size, lstm_hidden_size]
        # - lstm_state.c: cell state   [batch_size, lstm_hidden_size]
        init_lstm_state = tf_lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # construct customized LSTM network
        self.test, _, _, _ = self.recurrent_structure(batch_size, tf_lstm_cell, init_lstm_state)

    def recurrent_structure(self, 
            batch_size, 
            tf_lstm_cell,     # tensorflow LSTM cell object, e.g. 'tf.nn.rnn_cell.BasicLSTMCell'
            init_lstm_state): # initial LSTM state tensor
        """Recurrent structure with customized LSTM cells."""
        # defining initial data point
        # - init_t: initial time     [batch_size, t_dim] 
        init_t = tf.zeros([batch_size, self.t_dim], dtype=tf.float32)
        # concatenate each customized LSTM cell by loop
        seq_t = [] # generated sequence initialization
        seq_l = []
        seq_m = []
        last_t, last_lstm_state = init_t, init_lstm_state # loop initialization
        for _ in range(self.step_size):
            t, l, m, state = self._customized_lstm_cell(batch_size, tf_lstm_cell, last_lstm_state, last_t)
            seq_t.append(t)         # record generated time 
            seq_l.append(l)         # record generated location
            seq_m.append(m)         # record generated mark 
            last_t          = t     # reset last_t
            last_lstm_state = state # reset last_lstm_state
        seq_t = tf.stack(seq_t, axis=1)
        seq_l = tf.stack(seq_l, axis=1) 
        seq_m = tf.stack(seq_m, axis=1)
        return seq_t, seq_l, seq_m, state

    def _customized_lstm_cell(self, batch_size, tf_lstm_cell, last_state, t):
        """
        Customized Stochastic LSTM Cell

        The customized LSTM cell takes current (time 't', location 'l', mark 'm') and the hidden state of last moment
        as input, return the ('next_t', 'next_l', 'next_m') as well as the hidden state for the next moment. The time,
        location and mark will be sampled based upon last hidden state.

        The reason avoid using tensorflow builtin rnn structure is that, besides last hidden state, the other feedback 
        to next moment is a customized stochastic variable which depends on the last moment's rnn output. 
        """
        # stochastic neurons for generating time, location and mark
        next_t = t + self._delta_t(batch_size, last_state.h) # [batch_size, t_dim]
        next_l = self._l(batch_size, last_state.h) # [batch_size, 2] 
        next_m = self._m(batch_size, last_state.h) # [batch_size, m_dim] 
        x      = tf.concat([next_t, next_l], axis=1) # TODO: Add mark to input x
        # one step rnn structure
        # - x is a tensor that contains a single step of data points with shape [batch_size, t_dim + l_dim + m_dim]
        # - state is a tensor of hidden state with shape [2, batch_size, state_size]
        _, next_state = tf.nn.static_rnn(tf_lstm_cell, [x], initial_state=last_state, dtype=tf.float32)
        return next_t, next_l, next_m, next_state

    def _delta_t(self, batch_size, hidden_state):
        """Sampling time interval given hidden state of LSTM"""
        theta_h = tf.nn.elu(tf.matmul(hidden_state, self.Wt) + self.bt) + 1
        # reparameterization trick for sampling action from exponential distribution
        delta_t = - tf.log(tf.random_uniform([batch_size, self.t_dim], dtype=tf.float32)) / theta_h
        return delta_t

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
        l = tf.concat([x, y], axis=1)  # [batch_size, 2]
        # FOR LOGLIKELIHOOD
        # sigma1 = tf.sqrt(tf.square(sigma11) + tf.square(sigma12))
        # sigma2 = tf.sqrt(tf.square(sigma12) + tf.square(sigma22))
        # V12 = tf.multiply(sigma11, sigma12) + tf.multiply(sigma12, sigma22)
        # rho = V12 / tf.multiply(sigma1, sigma2)
        # z = tf.square(output_location_x - mu0) / tf.square(sigma1) \
        #     - 2 * tf.multiply(rho, tf.multiply(output_location_x - mu0, output_location_y - mu1)) / tf.multiply(sigma1,
        #                                                                                                         sigma2) \
        #     + tf.square(output_location_y - mu1) / tf.square(sigma2)
        # loglik = -z / 2 / (1 - tf.square(rho)) - tf.log(
        #     2 * pi * tf.multiply(tf.multiply(sigma1, sigma2), tf.sqrt(1 - tf.square(rho))))
        return l
    
    def _m(self, batch_size, hidden_state):
        """Sampling mark given hidden state of LSTM"""
        dense_feature = tf.nn.relu(tf.matmul(hidden_state, self.Wm0)) + self.bm0      # [batch_size, location_para_dim]
        dense_feature = tf.nn.elu(tf.matmul(dense_feature, self.Wm1) + self.bm1) + 1  # [batch_size, dim_m] dense_feature is positive
        # note that compressed_feature is not normalized yet
        # sample from multinomial distribution
        # (use Gumbel trick to sample the labels)
        eps        = 1e-13
        rv_uniform = tf.random_uniform([batch_size, self.m_dim])
        rv_Gumbel  = -tf.log(-tf.log(rv_uniform + eps) + eps)
        label      = tf.argmax(dense_feature + rv_Gumbel, axis=1) # label: [batch]
        m          = tf.one_hot(indices=label, depth=self.m_dim)
        return m

    def debug(self, sess):
        return sess.run(self.test)
        
if __name__ == "__main__":
    tf.set_random_seed(1)

    # Start training
    with tf.Session() as sess:

        step_size        = 7
        lstm_hidden_size = 5
        loc_hidden_size  = 10
        mak_hidden_size  = 10
        m_dim            = 3
        batch_size       = 6

        # initialize customized LSTM
        clstm = CustomizedStochasticLSTM(batch_size, step_size, lstm_hidden_size, loc_hidden_size, mak_hidden_size, m_dim)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        # Run the initializer
        sess.run(init)

        print(clstm.debug(sess))
        


# class PointProcessGenerator(object):
#     """
#     Point Process Generator
#     """

#     def __init__(self, T, seq_len, batch_size=2, state_size=1, kernel_bandwidth=1):
#         """

#         """
#         self.T = T
#         self.kernel_bandwidth = kernel_bandwidth 

#         self.state_size = state_size 
#         self.seq_len    = seq_len
#         self.batch_size = batch_size

#         # self.knock_out = 1 - tf.contrib.kfac.utils.kronecker_product(
#         #     tf.eye(2 * self.batch_size),
#         #     tf.ones([self.seq_len, self.seq_len]))
    
#     def construct_policy_network(self, hidden_size, batch_size):
#         """
#         Construct Policy Network
        
#         Policy should be flexible and expressive enough to capture the potential complex point process patterns of data.
#         Therefore, a customized recurrent neural network (RNN) with stochastic neurons is adopted, where hidden state is 
#         computed by hidden state of last moment and stochastically generated action. i.e.
#           a_{i+1} is sampling from pi(a|h_{i})
#           h_{i+1} = rnn_cell(h_{i}, a_{i+1})
#         """
#         pass

    # def reward(self, 
    #         expert_seq, 
    #         expert_seq_mask, 
    #         learner_seq, 
    #         learner_seq_mask, 
    #         loglik_interval_mask):

    #     # first compute mmd2 and then do all the reinforce trick.

    #     # first compute the entire kernel matrix in one shot

    #     # basis sequence: 
    #     basis_seq = tf.concat([expert_seq, learner_seq], axis=0)
    #     l2_kernel = tf.exp(-tf.square(basis_seq - tf.transpose(basis_seq)) / (self.kernel_bandwidth))
    #     # knock out all self interaction terms to get rid of the bias
    #     l2_kernel = tf.multiply(l2_kernel, self.knock_out)

    #     block_size = self.batch_size * self.seq_len
    #     input = tf.ones([self.seq_len, self.seq_len])
    #     # Lower triangular part
    #     reward_togo_mask = tf.contrib.kfac.utils.kronecker_product(
    #         tf.eye(self.batch_size),
    #         tf.matrix_band_part(input, -1, 0)
    #     )

    #     result = tf.zeros([1, self.batch_size * self.seq_len])


    #     # upper-right and lower-left block, contain one copy of the policy we are optimizing.
    #     result -= 2*tf.matmul(
    #         tf.multiply(expert_seq_mask, learner_seq_mask),
    #             l2_kernel[:block_size, block_size:])

    #     # lower-right block, contains two copies of the policy we are optimizing.
    #     # hence using the rule of total derivatives, we have 2 copies here.
    #     result += 2*tf.matmul(
    #         learner_seq_mask, l2_kernel[block_size:, block_size:])

    #     # compute reward to go
    #     result = tf.matmul(result, reward_togo_mask)
    #     result = tf.multiply(result,loglik_interval_mask)
    #     result = tf.reduce_sum(result)

    #     # upper-left block, does not contain the policy we are optimizing.
    #     # result += tf.matmul(
    #     # expert_time_mask,
    #     # tf.matmul(
    #     # 	norm2_kernel[:block_size, :block_size],
    #     # 	tf.transpose(expert_time_mask)
    #     # )
    #     # )

    #     print("reward v 6")

    #     # since sqrt is monotonic transformation, it can be removed in the optimization.
    #     return result / (self.batch_size * self.batch_size)

    #     # norm = tf.sqrt(norm2)
    #     # return norm

    # def compute_loglik_interval(self, learner_time_interval2, learner_time_mat_mask2, final_time_mask2):

    #     hidden_state = tf.zeros([self.batch_size, self.state_size], tf.float32)
    #     C_t = tf.zeros([self.batch_size, self.state_size], tf.float32)

    #     cum_output = tf.zeros([self.batch_size, 1])
    #     loglik_interval_array = []
    #     log_survival_array = []
    #     for i in range(self.seq_len):
    #         # the number 2 here is to compensate the 2 we put in the generator.
    #         # sigma = tf.exp(tf.matmul(hidden_state, self.V) + self.b2)
    #         sigma = (tf.nn.elu(tf.matmul(hidden_state, self.V) + self.b2) + 1)

    #         output = tf.reshape(learner_time_interval2[:,i], [self.batch_size, 1])
    #         # output_mask = tf.reshape(learner_time_mat_mask2[:,i], [self.batch_size, 1])

    #         loglik_interval = (tf.log(sigma) - sigma * output)
    #         loglik_interval_array.append(loglik_interval)
    #         # log_survival = -sigma * (self.T_max-cum_output)
    #         # log_survival_array.append(log_survival)

    #         cum_output += output

    #         aug_state = tf.concat([hidden_state, cum_output], axis=1)

    #         f_t = tf.nn.sigmoid(tf.matmul(aug_state, self.W_f) + self.b_f)
    #         i_t = tf.nn.sigmoid(tf.matmul(aug_state, self.W_i) + self.b_i)
    #         tC_t = tf.nn.tanh(tf.matmul(aug_state, self.W_C) + self.b_C)
    #         C_t = f_t * C_t + i_t * tC_t
    #         o_t = tf.nn.sigmoid(tf.matmul(aug_state, self.W_o) + self.b_o)

    #         hidden_state = o_t * tf.nn.tanh(C_t)

    #     # hidden_state = tf.nn.tanh(
    #         # 	tf.matmul(tf.concat([hidden_state, cum_output], axis=1), self.W)
    #         # 	+ self.b1
    #         # )

    #     # tmp_sum = tf.reduce_sum(
    #     # 	learner_time_mat_mask2 * tf.concat(loglik_interval_array, axis=1),
    #     # 	axis=1
    #     # )
    #     tmp = learner_time_mat_mask2 * tf.concat(loglik_interval_array, axis=1)
    #     # adding the survival term with mask, in general it seems that tf.Tensor object
    #     # works with masking but not direct indexing
    #     # tmp_survival = tf.reduce_sum(
    #     # 	final_time_mask2 * tf.concat(log_survival_array, axis=1),
    #     # 	axis=1
    #     # )
    #     # tmp_sum += tmp_survival

    #     print("new comp loglik 6")

    #     # since entire sequence gets the same log-likelihood term,
    #     # we replicate it to the entire sequence
    #     return tmp