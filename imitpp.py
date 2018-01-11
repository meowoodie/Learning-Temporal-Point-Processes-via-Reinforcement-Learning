#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Imitation Learning for Point Process


"""

import sys
import arrow
import random
import numpy as np
import tensorflow as tf

class PointProcessGenerater(object):
	"""

	For your information: 
	action has shape [batch_size, feature_size]
	state has shape  [batch_size, state_size]
	"""

	def __init__(self, seq_len=5, state_size=3, batch_size=2, feature_size=1):
		#TODO: remove seq_len
		tf.set_random_seed(100)
		self.seq_len      = seq_len
		self.batch_size   = batch_size
		self.feature_size = feature_size
		self.state_size   = state_size

		# create a BasicRNNCell
		self.rnn_cell = tf.contrib.rnn.BasicRNNCell(state_size)

		# input_data has shape [batch_size, sequence_len, feature_size]
		self.input_data = tf.placeholder(tf.float32, shape=[batch_size, seq_len, feature_size])

		# Optimizer of generator

	# Building Blocks of Computational Graph

	def _reward(self, t, expert_actions, max_learner_len=10):
		"""
		expert_actions has shape [batch_size, sequence_len]
		"""
		learner_actions, learner_states = self._fixed_length_rnn(self.batch_size, rnn_len=max_learner_len)
		# get time (index 0 = first element of a feature vector) of last action of each expert sequence
		expert_times    = expert_actions[:, :, 0]
		max_times       = tf.reduce_max(expert_times, reduction_indices=[1])
		# indices for action in batch of actions and 
		# indices for time in max times list
		action_indices  = tf.constant(np.array(range(self.batch_size)), tf.int32)
		# get learner times from learner actions
		learner_times   = tf.scan(
			# Only take index 0 value of feature vector into consideration
			# which means time of the action.
			lambda _, ind: tf.multiply(
				tf.cast(learner_actions[:, ind, 0] < max_times[ind], tf.float32), 
				learner_actions[:, ind, 0]),
			action_indices,
			initializer=np.zeros(self.seq_len, dtype=np.float32))

		return expert_times, learner_times

	def _fixed_length_rnn(self, num_seq, rnn_len):
		"""
		"""
		# state has shape [num_seq, state_size]
		initial_state = self.rnn_cell.zero_state(num_seq, dtype=tf.float32)
		# rnn always produce fixed length sequences
		# iteration counter for fixed length generator
		iter_counter = np.zeros(rnn_len)
		# iterate over counter to generate fixed length sequence
		actions, states = tf.scan(
			# action_state[0]: last action
			# action_state[1]: last state
			lambda action_state, _: self._dynamic_rnn_unit(num_seq, action_state[0], action_state[1]), 
			# used as counter in this loop
			iter_counter,
			# Initial action and state
			initializer=(np.zeros((num_seq, self.feature_size), dtype=np.float32), initial_state))
		# return a list of actions and a list of states that contain all the actions and states 
		# which are produced at each iteration step
		# actions has shape [rnn_len, num_seq=batch_size, feature_size]
		return actions, states
		
	def _dynamic_rnn_unit(self, num_seq, prv_action, prv_state):
		"""
		"""
		# reshape previous action to make it fit in the input of rnn
		# [num_seq, feature_size] -> [num_seq, 1, feature_size]
		rnn_input = tf.reshape(prv_action, [num_seq, 1, self.feature_size])

		_, cur_state = tf.nn.dynamic_rnn(self.rnn_cell, rnn_input,
										 initial_state=prv_state,
										 dtype=tf.float32)
		# Reparameterization trick
		# Reyleigh distribution with parameter sigma.
		W = tf.get_variable("W_r", [self.state_size, self.feature_size])
		b = tf.get_variable("b_r", [self.feature_size], initializer=tf.constant_initializer(0.0))
		sigma = tf.nn.elu(tf.matmul(cur_state, W) + b) + 1
		stoch_action = sigma * tf.sqrt(-2 * tf.log(tf.random_uniform(tf.shape(sigma), minval=0, maxval=1)))
		cur_action = tf.add(prv_action, sigma)
		# action has shape [num_seq, feature_size]
		# state has shape  [num_seq, state_size]
		return cur_action, cur_state

	# Available Functions

	def generate(self, sess, num_seq, max_t, max_learner_len=10, pretrained=False):
		"""
		"""
		actions, states = self._fixed_length_rnn(num_seq, rnn_len=max_learner_len)
		times = tf.multiply(tf.cast(actions[:, :, 0] < max_t, tf.float32), actions[:, :, 0])

		if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

		imit_times, states_history = sess.run([times, states])

		return imit_times, states_history

	def train(self, sess, expert_data, pretrained=False):
		"""
		"""
		#TODO: Move this to optimizer
		times, actions = self._reward(self.input_data, max_rnn_len=5)

		# Set pretrained variable if it was existed
		if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

		#TODO: Do training process in batch
		test_times, test_actions = sess.run([times, actions], feed_dict={
			self.input_data: [[[1], [1.5], [2], [0], [0]], [[1.1], [1.2], [2.4], [3.3], [5]], [[2], [2.2], [2.5], [3], [0]]]
			})



if __name__ == "__main__":
	seq_len      = 5
	batch_size   = 3
	state_size   = 4
	feature_size = 1

	with tf.Session() as sess:
		ppg = PointProcessGenerater(
			seq_len=seq_len,
			batch_size=batch_size, 
			state_size=state_size, 
			feature_size=feature_size)
		imit_times, states_history = ppg.generate(sess, num_seq=2, max_t=2)
		print imit_times




