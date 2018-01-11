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
	
	"""

	def __init__(self, seq_len=5, state_size=3, batch_size=2, feature_size=1):
		self.batch_size   = batch_size
		self.feature_size = feature_size
		self.state_size   = state_size

		# create a BasicRNNCell
		self.rnn_cell = tf.contrib.rnn.BasicRNNCell(state_size)

		# rnn_input has shape [batch_size, sequence_len, feature_size]
		self.input_data = tf.placeholder(tf.float32, shape=[batch_size, seq_len, feature_size])
		# state has shape [batch_size, state_size]
		self.initial_state = self.rnn_cell.zero_state(batch_size, dtype=tf.float32)

		# 1. Generator always produce fixed length sequences

		# iteration counter for fixed length generator
		iter_counter = np.zeros(seq_len)
		# final_state has shape [sequence_len, batch_size, state_size]
		self.flen_actions, self.flen_states = tf.scan(
			# last_res[0]: last action
			# last_res[1]: last state
			# x: element in input data
			lambda last_res, _: self._dynamic_rnn_unit(last_res[0], last_res[1]), 
			# used as counter in this loop
			iter_counter,
			# Initial rnn_output and state
			initializer=(np.zeros((self.batch_size, self.feature_size), dtype=np.float32), self.initial_state))

		# 2. Generator always produced fixed time frame sequences 

		max_t = tf.get_variable("max_t", [1])
		cond  = lambda t: t < max_t
		tf.while_loop()

	# Building Blocks of Computational Graph

	def _dynamic_rnn_unit(self, prv_action, prv_state):
		"""
		"""
		# reshape previous action to make it fit in the input of rnn
		# [batch_size, feature_size] -> [batch_size, 1, feature_size]
		rnn_input = tf.reshape(prv_action, [self.batch_size, 1, self.feature_size])

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

		return cur_action, cur_state	

	def _reward(self, expert_seqs, learner_seqs):
		"""
		"""

	# Available Functions

	def generate(self, sess, seq_len, pretrained=False):
		"""
		"""
		if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

		sess.run([self.actions, self.states], feed_dict={})


	def train(self, sess, expert_data, pretrained=False):
		"""
		"""
		# Set pretrained variable if it was existed
		if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

		#TODO: Do training process in batch





if __name__ == "__main__":

	batch_size   = 2
	state_size   = 3
	seq_len      = 5
	feature_size = 7

	tf.set_random_seed(100)

	# # rnn input has shape [batch_size, sequence_len, feature_size]
	# rnn_input = tf.placeholder(tf.float32, shape=[batch_size, seq_len, feature_size])

	# # create a BasicRNNCell
	# rnn_cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

	# # defining initial state
	# initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

	# # 'state' is a tensor of shape [batch_size, cell_state_size]
	# # 'outputs' is a tensor of shape [batch_size, sequence_len, cell_state_size]
	# rnn_outputs, state = tf.nn.dynamic_rnn(rnn_cell, rnn_input,
	#                                        initial_state=initial_state,
	#                                        dtype=tf.float32)

	with tf.Session() as sess:
		
		sess.run(tf.global_variables_initializer())

		# test_output, test_state = sess.run(
		# 	[rnn_outputs, state], 
		# 	feed_dict={rnn_input: [[[1],[2],[3],[4],[5]], [[5],[4],[3],[2],[1]]]})

		# imitpp = PointProcessGenerater(feature_size=feature_size)
		# sess.run(imitpp.init)
		# test_output, test_state = sess.run(
		# 	[imitpp.actions, imitpp.states],
		# 	feed_dict={imitpp.input_data: [
		# 	[[1,1,1,1,1,1,1],
		# 	 [2,2,2,2,2,2,2],
		# 	 [3,3,3,3,3,3,3],
		# 	 [4,4,4,4,4,4,4],
		# 	 [5,5,5,5,5,5,5]], 
		# 	[[1,1,1,1,1,1,1],
		# 	 [2,2,2,2,2,2,2],
		# 	 [3,3,3,3,3,3,3],
		# 	 [4,4,4,4,4,4,4],
		# 	 [5,5,5,5,5,5,5]]]})

		# # [5, 2, 1, 3]
		# print np.array(test_output)
		# # [5, 2, 3]
		# print np.array(test_state)

		max_t = 4
		cond = lambda t: t < max_t
		body = lambda t: tf.add(t, 1)
		r = tf.while_loop(cond, body, np.array([1,2,3,4,5]))
		res = sess.run(r)
		print res



	









