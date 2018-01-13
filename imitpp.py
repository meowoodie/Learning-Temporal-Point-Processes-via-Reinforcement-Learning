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

		#TODO: remove seq_len
		
		# tf.set_random_seed(100)

		self.seq_len      = seq_len
		self.batch_size   = batch_size
		self.feature_size = feature_size
		self.state_size   = state_size

		# create a BasicRNNCell
		self.rnn_cell = tf.contrib.rnn.BasicRNNCell(state_size)

		# input_data has shape [batch_size, sequence_len, feature_size]
		self.input_data = tf.placeholder(tf.float32, shape=[batch_size, seq_len, feature_size])

	# Building Blocks of Computational Graph

	def _reward(self, t, expert_actions, learner_actions):
		"""
		"""

		# get time (index 0 = first element of a feature vector) of last action of each expert sequence
		# expert_actions has shape [batch_size, sequence_len]
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

		# calculate kernel bandwidth
		medians = self._median_pairwise_distance(expert_times, learner_times)
		medians = tf.transpose(tf.reshape(tf.tile(medians, [self.seq_len]), (self.seq_len, self.batch_size)))
		# return reward
		reward = tf.reduce_mean(tf.subtract(
			tf.reduce_sum(self._gaussian_kernel(expert_times, t, medians), axis=1), \
			tf.reduce_sum(self._gaussian_kernel(learner_times, t, medians), axis=1)))

		return reward

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
		# here 1 means the length of input sequence for dynamic_rnn is 1 
		# because we repeatedly utilize only one step of dynamic_rnn for 
		# each iteraton to realize our customized rnn
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
		cur_action = tf.add(prv_action, stoch_action)
		# action has shape [num_seq, feature_size]
		# state has shape  [num_seq, state_size]
		return cur_action, cur_state

	def _median_pairwise_distance(self, seqAs, seqBs):
		"""
		reference: https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
		"""

		#TODO: remove the reshape process in the future. 
		#      for now, the seqs has shape [batch_size, seq_len],
		#      in the future, the calculation of distance could be applied to 
		#      high dimensional data, which means the input seqs has shape [batch_size, seq_len, dim]

		seq_len = tf.shape(seqAs)[1] + tf.shape(seqBs)[1]
		seqs = tf.concat([seqAs, seqBs], axis=1)
		seqs = tf.reshape(seqs, [self.batch_size, seq_len, 1])

		medians = tf.scan(
			lambda median, seq: self.__median(seq),
			seqs,
			# initializer=np.zeros(0, np.float32))
			initializer=np.array(0, dtype=np.float32))

		return medians

	def _gaussian_kernel(self, real_ts, var_ts, consts):
		"""
		"""
		# get mask of non zero value from real_t
		mask = tf.cast(real_ts > 0.0, dtype=tf.float32)
		# put mask on var_t in order to cancel the padding value in the real_t
		var_ts = tf.multiply(var_ts, mask)

		return tf.exp(- 1 * tf.divide(tf.square(real_ts - var_ts), (2 * consts)))

	@staticmethod
	def __median(seq):
		"""
		"""

		#TODO: mask should be handled carefully when dim is larger than 1. 
		#      a good mask will recognize the valid cells within seq. 
		#      simply using seq > 0 will not be able to be applied to high-dimensional cases.

		# mask is a matrix which consist of 1 and -1, means valid and invalid cells respectively.
		mask_col   = (tf.cast(seq > 0, dtype=tf.float32) - 0.5) * 2
		mask_row   = tf.transpose(mask_col)
		mask       = tf.matmul(mask_col, mask_row)
		valid_num  = tf.reduce_sum(tf.cast(mask >= 0, dtype=tf.int32))
		median_ind = tf.cast(valid_num/2, dtype=tf.int32)
		# calculate pairwise distance of a single seq
		r     = tf.reduce_sum(seq * seq, 1)
		r     = tf.reshape(r, [-1, 1])
		pdist = tf.reshape(   # flatten distanes into a 1D array
				tf.multiply(  # apply mask: remove invalid distances
					mask, (r - 2 * tf.matmul(seq, tf.transpose(seq)) + tf.transpose(r))), [-1]) 
		median = tf.nn.top_k(pdist, median_ind).values[median_ind-1] + \
			     tf.constant(1e-06, dtype=tf.float32) # in avoid of medians is 0

		return median

	# Available Functions
	# Runnable Computational Graphs

	def generate(self, sess, num_seq, max_t, max_learner_len=10, pretrained=False):
		"""
		"""

		# Generator Computational Graph

		actions, states = self._fixed_length_rnn(num_seq, rnn_len=max_learner_len)
		times = tf.multiply(tf.cast(actions[:, :, 0] < max_t, tf.float32), actions[:, :, 0])

		# Runnning Graph

		if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

		imit_times, states_history = sess.run([times, states])

		return imit_times, states_history

	def train(self, sess, input_data, test_data, iters=10, display_step=1, pretrained=False):
		"""
		"""

		# Optimizer Computational Graph
		# notes: putting optimizer graph outside of the init function since it would cause 
		#        a conflict when you try to run the generate graph. Because the tensorflow
		#        will try to compile a _fixed_length_rnn subgraph which has been initialized
		#        in optimizer graph.

		# expert actions and corresponding learner actions
		# expert_actions has shape [batch_size, seq_len, feature_size]
		# learner_actions has shape [seq_len, batch_size, feature_size]
		expert_actions     = self.input_data #TODO: replace input_data with a batch of input_data
		learner_actions, _ = self._fixed_length_rnn(self.batch_size, rnn_len=self.seq_len)
		# in order to avoid a specific bug (or flaw) triggerred by tensorflow itself
		# here unfold matrix into a 1D list first, then apply reward function to every single element
		# in the tensor, finally, refold the result back to a matrix with same shape with original one
		unfold_times = tf.reshape(learner_actions[:, :, 0], [-1])
		unfold_rewards = tf.map_fn(lambda t: self._reward(t, expert_actions, learner_actions), unfold_times)
		refold_rewards = tf.reshape(unfold_rewards, [self.seq_len, self.batch_size])
		# # loss function
		self.loss = tf.reduce_mean(tf.reduce_sum(refold_rewards, axis=0))
		# # optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.loss)

		# Runnning Graph

		# Set pretrained variable if it was existed
		if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

		step        = 1 # the step of the iteration
		start_index = 0 # the index of the start row of the batch
		# Keep training until reach max iterations
		while step * self.batch_size <= iters:
			# Fetch the next batch of the input data (q, d, y)
			# And update the start_indext in order to prepare for the next batch
			batch_input_data, start_index = self._next_batch(input_data, start_index)
			# Run optimization
			sess.run(self.optimizer, feed_dict={self.input_data: batch_input_data})
			if step % display_step == 0:
				# Calculate batch loss and accuracy
				train_loss = sess.run(self.loss, feed_dict={self.input_data: batch_input_data})
				test_loss  = sess.run(self.loss, feed_dict={self.input_data: test_data})
				# Log information for each iteration
				print >> sys.stderr, "[%s] Iter: %d" % (arrow.now(), (step * self.batch_size)) 
				print >> sys.stderr, "[%s] Train Loss: %.5f,\tTest Loss: %.5f" % (arrow.now(), train_loss, test_loss)
			step += 1
		print >> sys.stderr, "[%s] Optimization Finished!" % arrow.now()

	def _next_batch(self, input_data, start_index):
		"""
		Next Batch
		
		This is a private method for fetching a batch of data from the integral input data. 
		Each time you call this method, it would return the next batch in the dataset by indicating 
		the start index. Which means you have to keep the return start index of every invoking,
		and pass it to the next invoking for getting the next batch correctly. 
		"""

		# total number of rows of the input data (query and target)
		num_seq, num_action, num_feature = np.shape(input_data)   
		# start index of the row of the input data
		start_seq = start_index % num_seq 
		# end index of the row of the input data
		end_seq   = (start_seq + self.batch_size) % num_seq 
		# if there is not enought data left in the dataset for generating an integral batch,
		# then top up this batch by traversing back to the start of the dataset. 
		if end_seq < start_seq:
			batch_input_data = np.append(input_data[start_seq: num_seq], input_data[0: end_seq], axis=0).astype(np.float32)
		else:
			batch_input_data = input_data[start_seq: end_seq].astype(np.float32)
		# Update the start index
		start_index += self.batch_size
		return batch_input_data, start_index