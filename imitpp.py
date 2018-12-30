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

class StochasticLSTM(object):
    """
    Customized Stochastic LSTM Network
    """

    def __init__(self, step_size, hidden_size, batch_size):
        self.batch_size  = batch_size
        self.step_size   = step_size
        self.hidden_size = hidden_size

    def recurrent_structure(self):
        # create a basic LSTM cell
        tf_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        # defining initial basic LSTM hidden state
        init_state = tf_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # loop for each step in LSTM
        for i in range(self.step_size):
            self.customized_lstm_cell(tf_lstm_cell, init_state)

    def customized_lstm_cell(self, tf_rnn_cell, x, last_state):
        """
        Customized LSTM Cell

        The reason avoid using tensorflow builtin rnn structure is that, besides last hidden state, the other feedback 
        to next moment is a customized stochastic variable which depends on the last moment's rnn output. 
        """
        # 'x' is a tensor that contains a single step of data points with shape [batch_size, data_dim]
        # 'state' is a tensor of shape [batch_size, cell_state_size]
        outputs, state = tf.nn.static_rnn(tf_rnn_cell, x, initial_state=last_state, dtype=tf.float32)
        # stochastic neurons
        theta_h = tf.nn.elu(tf.matmul(state, self.W) + self.b) + 1
        # reparameterization trick for sampling action from exponential distribution
        output  = - np.log(np.random.rand(self.batch_size).astype(np.float32)) / theta_h
        




class PointProcessGenerator(object):
    """
    Point Process Generator
    """

    def __init__(self, T, seq_len, batch_size=2, state_size=1, kernel_bandwidth=1):
        """

        """
        self.T = T
        self.kernel_bandwidth = kernel_bandwidth 

        self.state_size = state_size 
        self.seq_len    = seq_len
        self.batch_size = batch_size

        # self.knock_out = 1 - tf.contrib.kfac.utils.kronecker_product(
        #     tf.eye(2 * self.batch_size),
        #     tf.ones([self.seq_len, self.seq_len]))
    
    def construct_policy_network(self, hidden_size, batch_size):
        """
        Construct Policy Network
        
        Policy should be flexible and expressive enough to capture the potential complex point process patterns of data.
        Therefore, a customized recurrent neural network (RNN) with stochastic neurons is adopted, where hidden state is 
        computed by hidden state of last moment and stochastically generated action. i.e.
          a_{i+1} is sampling from pi(a|h_{i})
          h_{i+1} = rnn_cell(h_{i}, a_{i+1})
        """
        
        
        


            

        

    def reward(self, 
            expert_seq, 
            expert_seq_mask, 
            learner_seq, 
            learner_seq_mask, 
            loglik_interval_mask):

        # first compute mmd2 and then do all the reinforce trick.

        # first compute the entire kernel matrix in one shot

        # basis sequence: 
        basis_seq = tf.concat([expert_seq, learner_seq], axis=0)
        l2_kernel = tf.exp(-tf.square(basis_seq - tf.transpose(basis_seq)) / (self.kernel_bandwidth))
        # knock out all self interaction terms to get rid of the bias
        l2_kernel = tf.multiply(l2_kernel, self.knock_out)

        block_size = self.batch_size * self.seq_len
        input = tf.ones([self.seq_len, self.seq_len])
        # Lower triangular part
        reward_togo_mask = tf.contrib.kfac.utils.kronecker_product(
            tf.eye(self.batch_size),
            tf.matrix_band_part(input, -1, 0)
        )

        result = tf.zeros([1, self.batch_size * self.seq_len])


        # upper-right and lower-left block, contain one copy of the policy we are optimizing.
        result -= 2*tf.matmul(
            tf.multiply(expert_seq_mask, learner_seq_mask),
                l2_kernel[:block_size, block_size:])

        # lower-right block, contains two copies of the policy we are optimizing.
        # hence using the rule of total derivatives, we have 2 copies here.
        result += 2*tf.matmul(
            learner_seq_mask, l2_kernel[block_size:, block_size:])

        # compute reward to go
        result = tf.matmul(result, reward_togo_mask)
        result = tf.multiply(result,loglik_interval_mask)
        result = tf.reduce_sum(result)

        # upper-left block, does not contain the policy we are optimizing.
        # result += tf.matmul(
        # expert_time_mask,
        # tf.matmul(
        # 	norm2_kernel[:block_size, :block_size],
        # 	tf.transpose(expert_time_mask)
        # )
        # )

        print("reward v 6")

        # since sqrt is monotonic transformation, it can be removed in the optimization.
        return result / (self.batch_size * self.batch_size)

        # norm = tf.sqrt(norm2)
        # return norm

    def compute_loglik_interval(self, learner_time_interval2, learner_time_mat_mask2, final_time_mask2):

        hidden_state = tf.zeros([self.batch_size, self.state_size], tf.float32)
        C_t = tf.zeros([self.batch_size, self.state_size], tf.float32)

        cum_output = tf.zeros([self.batch_size, 1])
        loglik_interval_array = []
        log_survival_array = []
        for i in range(self.seq_len):
            # the number 2 here is to compensate the 2 we put in the generator.
            # sigma = tf.exp(tf.matmul(hidden_state, self.V) + self.b2)
            sigma = (tf.nn.elu(tf.matmul(hidden_state, self.V) + self.b2) + 1)

            output = tf.reshape(learner_time_interval2[:,i], [self.batch_size, 1])
            # output_mask = tf.reshape(learner_time_mat_mask2[:,i], [self.batch_size, 1])

            loglik_interval = (tf.log(sigma) - sigma * output)
            loglik_interval_array.append(loglik_interval)
            # log_survival = -sigma * (self.T_max-cum_output)
            # log_survival_array.append(log_survival)

            cum_output += output

            aug_state = tf.concat([hidden_state, cum_output], axis=1)

            f_t = tf.nn.sigmoid(tf.matmul(aug_state, self.W_f) + self.b_f)
            i_t = tf.nn.sigmoid(tf.matmul(aug_state, self.W_i) + self.b_i)
            tC_t = tf.nn.tanh(tf.matmul(aug_state, self.W_C) + self.b_C)
            C_t = f_t * C_t + i_t * tC_t
            o_t = tf.nn.sigmoid(tf.matmul(aug_state, self.W_o) + self.b_o)

            hidden_state = o_t * tf.nn.tanh(C_t)

        # hidden_state = tf.nn.tanh(
            # 	tf.matmul(tf.concat([hidden_state, cum_output], axis=1), self.W)
            # 	+ self.b1
            # )

        # tmp_sum = tf.reduce_sum(
        # 	learner_time_mat_mask2 * tf.concat(loglik_interval_array, axis=1),
        # 	axis=1
        # )
        tmp = learner_time_mat_mask2 * tf.concat(loglik_interval_array, axis=1)
        # adding the survival term with mask, in general it seems that tf.Tensor object
        # works with masking but not direct indexing
        # tmp_survival = tf.reduce_sum(
        # 	final_time_mask2 * tf.concat(log_survival_array, axis=1),
        # 	axis=1
        # )
        # tmp_sum += tmp_survival

        print("new comp loglik 6")

        # since entire sequence gets the same log-likelihood term,
        # we replicate it to the entire sequence
        return tmp