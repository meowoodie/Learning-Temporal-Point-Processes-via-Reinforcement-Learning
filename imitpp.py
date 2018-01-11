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

class ImitPP(object):
	"""
	
	"""

	def __init__(self, seq_len=5, state_size=3, batch_size=2, feature_size=1):
		self.batch_size   = batch_size
		self.feature_size = feature_size

		# create a BasicRNNCell
		self.rnn_cell = tf.contrib.rnn.BasicRNNCell(state_size)

		# rnn_input has shape [batch_size, sequence_len, feature_size]
		self.input_data = tf.placeholder(tf.float32, shape=[batch_size, seq_len, feature_size])
		# transpose input to a sequence which is able to fit in scan loop
		# seqs has shape [sequence_len, batch_size, feature_size]
		seqs = tf.transpose(self.input_data, perm=[1, 0, 2])
		# state has shape [batch_size, state_size]
		initial_state = self.rnn_cell.zero_state(batch_size, dtype=tf.float32)
		
		# final_output has shape [batch_size, sequence_len=1, feature_size]
		self.final_state = tf.scan(
			# a[0]: rnn_outputs
			# a[1]: state
			# x: element in input data
			lambda a, x: self.dynamic_rnn_unit(x, a), 
			# iterate the element [batch_size, feature_size] of sequence in input data
			seqs,
			# Initial rnn_output and state
			initializer=initial_state)

		self.init = tf.global_variables_initializer()



	def dynamic_rnn_unit(self, rnn_input, prv_state):

		rnn_input = tf.reshape(rnn_input, [self.batch_size, 1, self.feature_size])

		# rnn_outputs has shape [batch_size, sequence_len=1, feature_size]
		rnn_outputs, cur_state = tf.nn.dynamic_rnn(self.rnn_cell, rnn_input,
												   initial_state=prv_state,
												   dtype=tf.float32)
		# Reparameterization trick
        # Reyleigh distribution with parameter sigma.
		# W = tf.get_variable("W_r", [state_size, 1])
  #       b = tf.get_variable("b_r", [1], initializer=tf.constant_initializer(0.0))
  #       sigma  = tf.nn.elu(tf.matmul(current_action, W) + b) + 1
		# action = sigma * tf.sqrt(-2 * tf.random_uniform( tf.shape(sigma), minval=0, maxval=1) )


		return cur_state


if __name__ == "__main__":

	batch_size   = 2
	state_size   = 3
	seq_len      = 5
	feature_size = 1

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
		
		# sess.run(tf.global_variables_initializer())

		# test_output, test_state = sess.run(
		# 	[rnn_outputs, state], 
		# 	feed_dict={rnn_input: [[[1],[2],[3],[4],[5]], [[5],[4],[3],[2],[1]]]})

		imitpp = ImitPP()
		sess.run(imitpp.init)
		test_state = sess.run(
			[imitpp.final_state],
			feed_dict={imitpp.input_data: [[[1],[2],[3],[4],[5]], [[5],[4],[3],[2],[1]]]})

		print "test state"
		print test_state

	









