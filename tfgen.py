#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Imitation Learning for Point Process

A LSTM based model for generating marked spatial-temporal points.

References:
- https://arxiv.org/abs/1811.05016

Dependencies:
- Python 3.6.7
- tensorflow==1.5.0
"""

import sys
import arrow
import numpy as np
import tensorflow as tf

from stppg import DiffusionKernel, HawkesLam, MarkedSpatialTemporalPointProcess

class MarkedSpatialTemporalHawkes(object):
    """
    """

    def __init__(self):
        """
        """
        self.mu      = tf.get_variable(name="mu", initializer=tf.random_uniform(shape=(), minval=0, maxval=1))
        self.beta    = tf.get_variable(name="beta", initializer=tf.random_uniform(shape=(), minval=0, maxval=1))
        self.sigma_x = tf.get_variable(name="sigma_x", initializer=tf.random_uniform(shape=(), minval=0, maxval=1))
        self.sigma_y = tf.get_variable(name="sigma_y", initializer=tf.random_uniform(shape=(), minval=0, maxval=1))

    def _sampling(self, sess, T, S, batch_size):
        """
        """
        # get current model parameters
        mu, beta, sigma_x, sigma_y = sess.run([self.mu, self.beta, self.sigma_x, self.sigma_y])
        # sampling points given model parameters
        kernel = DiffusionKernel(beta=beta, sigma_x=sigma_x, sigma_y=sigma_y)
        lam    = HawkesLam(mu, kernel, maximum=1e+6)
        pp     = MarkedSpatialTemporalPointProcess(lam)
        return pp.generate(T=T, S=S, batch_size=batch_size)

    # def _log_likelihood(self, data):
    #     """
    #     """



class MarkedSpatialTemporalLSTM(object):
    """
    Customized Stochastic LSTM Network

    A LSTM Network with customized stochastic output neurons, which used to generate time, location and marks accordingly.
    """

    def __init__(self, step_size, lstm_hidden_size, loc_hidden_size, mak_hidden_size, m_dim, x_lim=5, y_lim=5, epsilon=0.3):
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
        self.step_size         = step_size        # step size of LSTM
        self.lstm_hidden_size  = lstm_hidden_size # size of LSTM hidden feature
        self.loc_hidden_size   = loc_hidden_size  # size of location hidden feature
        self.loc_param_size    = 5                # by default
        self.mak_hidden_size   = mak_hidden_size  # size of mark hidden feature
        self.x_lim, self.y_lim = x_lim, y_lim
        self.epsilon           = epsilon

        INIT_PARAM_RATIO = 1 / np.sqrt(self.loc_hidden_size * self.loc_param_size)

        # define learning weights
        # - time weights
        self.Wt  = tf.get_variable(name="Wt", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.t_dim]))
        self.bt  = tf.get_variable(name="bt", initializer=INIT_PARAM_RATIO * tf.random_normal([self.t_dim]))
        # - location weights
        self.Wl0 = tf.get_variable(name="Wl0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.loc_hidden_size]))
        self.bl0 = tf.get_variable(name="bl0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.loc_hidden_size]))
        self.Wl1 = tf.get_variable(name="Wl1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.loc_hidden_size, self.loc_param_size]))
        self.bl1 = tf.get_variable(name="bl1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.loc_param_size]))
        # - mark weights
        self.Wm0 = tf.get_variable(name="Wm0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.mak_hidden_size]))
        self.bm0 = tf.get_variable(name="bm0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.mak_hidden_size]))
        self.Wm1 = tf.get_variable(name="Wm1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.mak_hidden_size, self.m_dim]))
        self.bm1 = tf.get_variable(name="bm1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.m_dim]))

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
        loglik = loglik_l # + loglik_l # + loglik_m    # TODO: Add mark to input x
        # input of LSTM
        x      = tf.concat([next_l], axis=1) # TODO: Add mark to input x
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
        # masks for epsilon greedy exploration & regular sampling
        p = tf.random_uniform([batch_size, 1], 0, 1)                  # [batch_size, 1]
        l_eps_mask = tf.cast(p < self.epsilon, dtype=tf.float32)      # [batch_size, 1]
        l_reg_mask = 1. - l_eps_mask                                  # [batch_size, 1]

        # sample from uniform distribution (epsilon greedy exploration)
        lx_eps = tf.random_uniform([batch_size, 1], minval=-self.x_lim, maxval=self.x_lim, dtype=tf.float32)
        ly_eps = tf.random_uniform([batch_size, 1], minval=-self.y_lim, maxval=self.y_lim, dtype=tf.float32)

        # sample from the distribution detemined by hidden state
        dense_feature = tf.nn.relu(tf.matmul(hidden_state, self.Wl0)) + self.bl0  # [batch_size, loc_hidden_size]
        dense_feature = tf.matmul(dense_feature, self.Wl1) + self.bl1             # [batch_size, loc_param_size]
        # - 5 params that determine the distribution of location shifts with shape [batch_size]
        mu0 = tf.reshape(dense_feature[:, 0], [batch_size, 1]) 
        mu1 = tf.reshape(dense_feature[:, 1], [batch_size, 1])
        # - construct positive definite and symmetrical matrix as covariance matrix
        A11 = tf.expand_dims(tf.reshape(dense_feature[:, 2], [batch_size, 1]), -1) # [batch_size, 1, 1]
        A22 = tf.expand_dims(tf.reshape(dense_feature[:, 3], [batch_size, 1]), -1) # [batch_size, 1, 1]
        A21 = tf.expand_dims(tf.reshape(dense_feature[:, 4], [batch_size, 1]), -1) # [batch_size, 1, 1]
        A12 = tf.zeros([batch_size, 1, 1])                                         # [batch_size, 1, 1]
        A1  = tf.concat([A11, A12], axis=2) # [batch_size, 1, 2]
        A2  = tf.concat([A21, A22], axis=2) # [batch_size, 1, 2]
        A   = tf.concat([A1, A2], axis=1)   # [batch_size, 2, 2]
        # - sigma = A * A^T with shape [batch_size, 2, 2]
        sigma   = tf.scan(lambda a, x: tf.matmul(x, tf.transpose(x)), A) # [batch_size, 2, 2]
        sigma11 = tf.expand_dims(sigma[:, 0, 0], -1)                     # [batch_size, 1]
        sigma22 = tf.expand_dims(sigma[:, 1, 1], -1)                     # [batch_size, 1]
        sigma12 = tf.expand_dims(sigma[:, 0, 1], -1)                     # [batch_size, 1]
        # - random variable for generating locaiton
        rv0 = tf.random_normal([batch_size, 1])
        rv1 = tf.random_normal([batch_size, 1])
        # - location x and y
        x = mu0 + tf.multiply(sigma11, rv0) + tf.multiply(sigma12, rv1) # [batch_size, 1]
        y = mu1 + tf.multiply(sigma12, rv0) + tf.multiply(sigma22, rv1) # [batch_size, 1]

        # # combine exploration and regular sampling
        # x = tf.multiply(lx_eps, l_eps_mask) + tf.multiply(x, l_reg_mask)
        # y = tf.multiply(ly_eps, l_eps_mask) + tf.multiply(y, l_reg_mask)
        l = tf.concat([x, y], axis=1)                         # [batch_size, 2]

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

if __name__ == "__main__":
    # training model
    tf.set_random_seed(1)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        hawkes = MarkedSpatialTemporalHawkes()
        points = hawkes._sampling(sess, T=[0., 1.], S=[[-1., 1.], [-1., 1.]], batch_size=3)
        print(points)
