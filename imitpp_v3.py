#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kernel script for Imitation Learning for Point Process, mainly including class
PointProcessGenerator

[tensorflow==1.3.0]
tensorflow higher than 1.3.0 does not support get gradiant for loop structure of
RNN unit.
"""

import sys
import arrow
import random
import numpy as np
import tensorflow as tf

from utils.plots import *

class PointProcessGenerator(object):
	"""
	Point Process Generator is a highly customized RNN model for generating
	point process actions by imitating input expert actions.
	Param:
	- t_max:        maximum time of expert actions
	- seq_len:      length of sequence (with zero padding) of actions
	- state_size:   size of hidden state of RNN
	- batch_size:   size of batch data for training
	- feature_size: size of feature (including time at index 0)
	- iters:        iterations of training = epoche (step) * batch_size
	- display_step: log training information at display_step
	- lr:           learning rate
	"""

	def __init__(self, t_max, seq_len,
	             state_size=3, batch_size=2, feature_size=1,
				 iters=10, display_step=1, lr=1e-4):

		tf.set_random_seed(100)
		self.seq_len      = seq_len
		self.batch_size   = batch_size
		self.feature_size = feature_size
		self.state_size   = state_size
		self.iters        = iters
		self.display_step = display_step

		# Sigma parameters
		self.W_r = tf.get_variable("W_r", [self.state_size, self.feature_size])
		self.b_r = tf.get_variable("b_r", [self.feature_size], initializer=tf.constant_initializer(0.0))

		# create a BasicRNNCell
		self.rnn_cell = tf.contrib.rnn.LSTMCell(state_size)
		# self.rnn_cell = tf.contrib.rnn.BasicRNNCell(state_size)

		# input_data has shape [batch_size, sequence_len, feature_size]
		self.input_data = tf.placeholder(tf.float32, shape=[batch_size, seq_len, feature_size])
		self.t_max      = t_max # tf.constant(t_max, dtype=tf.float32)

		# Optimizer Computational Graph
		# expert actions and corresponding learner actions
		# expert_actions has shape [batch_size, seq_len, feature_size]
		# the orginal learner_actions has shape [seq_len, batch_size, feature_size]
		expert_actions                  = self.input_data
		learner_actions, states, sigmas = self._fixed_length_rnn(self.batch_size, rnn_len=self.seq_len)
		# learner_actions has shape [batch_size, seq_len, feature_size]
		# which now has the same shape as expert_actions
		learner_actions = tf.transpose(learner_actions, perm=(1, 0, 2))

		# Loglikelihood
		loglik    = self._loglik_interval(learner_actions[:, :, 0], states, sigmas)
		# loss function
		self.loss = self._reward_loss(expert_actions, learner_actions, loglik)
		# optimizer
		# learning_rate  = tf.train.exponential_decay(lr, 50, 0.96, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9)\
		                   .minimize(self.loss)
		# self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)\
		#                    .minimize(self.loss)

		# # Generator Computational Graph
		# self.expert_times  = expert_actions[:, :, 0]
		self.learner_times = tf.multiply(tf.cast(learner_actions[:, :, 0] < self.t_max, tf.float32), learner_actions[:, :, 0])
		# self.learner_times = learner_actions[:, :, 0]
		# self.states        = states
		# self.sigmas        = sigmas

	# Building Blocks of Computational Graph

	def _reward_loss(self, expert_actions, learner_actions, loglik):
		"""
		Reward Loss
		"""
		# TODO: here we only consider the time in the action feature. In the future
		#       other feature will be taken into account.
		unfolded_learner_times = tf.reshape(learner_actions, [self.batch_size*self.seq_len, 1])
		unfolded_expert_times  = tf.reshape(expert_actions, [self.batch_size*self.seq_len, 1])
		# Prepare masks for learner_time and expert_time (filter invalid "< max_t" values)
		learner_times_mask     = tf.cast(unfolded_learner_times < self.t_max, dtype=tf.float32)
		expert_times_mask      = tf.cast(unfolded_expert_times > 0.0, dtype=tf.float32)
		# unfolded_learner_times = tf.multiply(unfolded_learner_times, learner_times_mask)
		# Concatenate expert_time and learner_time and prepare their mask for fast matrix operation
		basis_times      = tf.concat([unfolded_expert_times, unfolded_learner_times], axis=0)
		basis_times_mask = tf.transpose(tf.concat([expert_times_mask, -1 * learner_times_mask], axis=0))
		# Calculate the kernel bandwidth, which is the median value of expert_time and learner_time
		# kernel_bandwidth = self._median_pairwise_distance(unfolded_expert_times, unfolded_learner_times)

		# knock out all self interaction terms to get rid of the bias
		norm2_kernel = tf.exp(-tf.square(basis_times - tf.transpose(basis_times)) / 0.5) # kernel_bandwidth)
		norm2_kernel = tf.multiply(norm2_kernel, self._kronecker_product())

		# upper-right and lower-left block, contain one copy of the policy we are optimizing.
		block_size = self.batch_size * self.seq_len
		reward = tf.matmul(tf.transpose(expert_times_mask),
			tf.matmul(
				norm2_kernel[:block_size, block_size:],
				tf.transpose(tf.transpose(learner_times_mask) * loglik)))

		# lower-right block, contains two copies of the policy we are optimizing.
		# hence using the rule of total derivatives, we have 2 copies here.
		reward += tf.matmul(
			tf.transpose(learner_times_mask),
			tf.matmul(
				norm2_kernel[block_size:, block_size:],
				tf.transpose(tf.transpose(learner_times_mask) * loglik)))

		return reward / (self.batch_size * self.batch_size)

	def _loglik_interval(self, learner_times, states, sigmas):
		"""
		learner_times: [batch_size, seq_len]
		states:        [seq_len, batch_size, state_size]
		sigmas:        [seq_len, batch_size, feature_size]
		"""
		# sigmas has shape [batch_size, seq_len]
		sigmas = tf.transpose(sigmas[:, :, 0])
		# learner_times_diff has shape [batch_size, seq_len]
		learner_times_diff = tf.subtract(learner_times[:, 1:], learner_times[:, :-1])
		learner_times_diff = tf.concat([
			tf.reshape(learner_times[:, 0], [self.batch_size, 1]),
			learner_times_diff], axis=1)
		# loglik_interval has shape [batch_size, seq_len]
		loglik_interval = tf.log(sigmas) - tf.multiply(sigmas, learner_times_diff)
		# log_survival has shape [batch_size, seq_len]
		log_survival    = -1 * tf.multiply(sigmas, (self.t_max - learner_times))

		learner_times_mask  = tf.cast(learner_times < self.t_max, dtype=tf.float32)
		loglik_interval_sum = tf.reduce_sum(tf.multiply(learner_times_mask, loglik_interval), axis=1)
		# tf.reduce_sum(learner_times_mask, axis=1) - 1
		# log_survival_max    = tf.reduce_max(tf.multiply(learner_times_mask, log_survival), axis=1)
		# return [tf.reduce_sum(learner_times_mask, axis=1) - 1]
		return tf.reshape(
			tf.reshape(loglik_interval_sum, [self.batch_size, 1]) * tf.ones([1,self.seq_len]),
			[1, self.batch_size * self.seq_len])

	def _fixed_length_rnn(self, num_seq, rnn_len):
		"""
		Fixed Length RNN implemented by dynamic_rnn of tensorflow
		"""
		# state has shape [num_seq, state_size]
		initial_action = np.zeros((num_seq, self.feature_size), dtype=np.float32)
		initial_state  = self.rnn_cell.zero_state(num_seq, dtype=tf.float32)
		initial_sigma  = np.zeros((self.batch_size, self.feature_size), dtype=np.float32)
		# rnn always produce fixed length sequences
		# iteration counter for fixed length generator
		iter_counter = np.zeros(rnn_len)
		# iterate over counter to generate fixed length sequence
		actions, states, sigmas = tf.scan(
			# action_state[0]: last action
			# action_state[1]: last state
			lambda action_state_sigma, _: self._dynamic_rnn_unit(num_seq, action_state_sigma[0], action_state_sigma[1]),
			# used as counter in this loop
			iter_counter,
			# Initial action and state
			initializer=(initial_action, initial_state, initial_sigma))
		# return a list of actions and a list of states that contain all the actions and states
		# which are produced at each iteration step
		# actions has shape [rnn_len, num_seq=batch_size, feature_size]
		return actions, states, sigmas

	def _dynamic_rnn_unit(self, num_seq, prv_action, prv_state):
		"""
		One step in dynamic_rnn of tensorflow
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
		# W = tf.get_variable("W_r", [self.state_size, self.feature_size])
		# b = tf.get_variable("b_r", [self.feature_size], initializer=tf.constant_initializer(0.0))
		cur_sigma    = tf.exp(tf.matmul(cur_state.h, self.W_r) + self.b_r)
		stoch_action = -1 * tf.log(tf.random_uniform(tf.shape(cur_sigma), minval=0, maxval=1)) / cur_sigma
		cur_action   = tf.add(prv_action, stoch_action)
		# action has shape [num_seq, feature_size]
		# state has shape  [num_seq, state_size]
		return cur_action, cur_state, cur_sigma

	def _kronecker_product(self):
		"""
		Compute Kronecker Product

		Since tensorflow == 1.3.0 does not support tf.contrib.kfac.utils.kronecker_product
		This function manully calculate kronecker product by indicating batch_size
		and seq_len
		"""
		# Full matrix
		full = tf.zeros((0, 2 * self.batch_size * self.seq_len))
		# Indicating matrix for locating the zeros sub matrix
		locs = np.eye(2 * self.batch_size)
		# Constructing full matrix with zeros and ones matrix pads
		for i in range(2 * self.batch_size):
		    row = tf.zeros((self.seq_len, 0))
		    for j in range(2 * self.batch_size):
		        if locs[i, j] == 1.:
		            pad = tf.zeros([self.seq_len, self.seq_len])
		        else:
		            pad = tf.ones([self.seq_len, self.seq_len])
		        row = tf.concat([row, pad], axis=1)
		    full = tf.concat([full, row], axis=0)
		return full

	def _median_pairwise_distance(self, seqAs, seqBs):
		"""
		Get median value of pairwise distances for two tensors.
		"""
		#TODO: remove the reshape process in the future.
		#      for now, the seqs has shape [batch_size, seq_len],
		#      in the future, the calculation of distance could be applied to
		#      high dimensional data, which means the input seqs has shape [batch_size, seq_len, dim]
		seqAs = tf.reshape(seqAs, [-1])
		seqBs = tf.reshape(seqBs, [-1])
		seq   = tf.concat([seqAs, seqBs], axis=0)
		seq   = tf.reshape(seq, [tf.size(seq), 1])
		return self.__median(seq)

	@staticmethod
	def __median(seq):
		"""
		Calculate median value of pairwise distance between two arbitrary points
		in a sequence of points.
		Reference for calculating median value of a sequence:
		https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
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
		median = tf.nn.top_k(pdist, median_ind).values[median_ind - 1] + \
			     tf.constant(1e-06, dtype=tf.float32) # in avoid of medians is 0
		return median

	# Available Functions
	# Runnable Computational Graphs

	def generate(self, sess, pretrained=False):
		"""
		Generate actions
		"""
		if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

		imit_times, states_history = sess.run([self.learner_times, self.states])

		return imit_times, states_history

	def train(self, sess, input_data, test_data=None, pretrained=False):
		"""
		Train model
		"""
		# Set pretrained variable if it was existed
		if not pretrained:
			init = tf.global_variables_initializer()
			sess.run(init)

		step        = 1 # the step of the iteration
		start_index = 0 # the index of the start row of the batch
		# Keep training until reach max iterations
		while step * self.batch_size <= self.iters:
			# Fetch the next batch of the input data (q, d, y)
			# And update the start_indext in order to prepare for the next batch
			batch_input_data, start_index = self._next_batch(input_data, start_index)
			# Run optimization
			sess.run(self.optimizer, feed_dict={self.input_data: batch_input_data})
			if step % self.display_step == 0:
				# Calculate batch loss and accuracy
				train_loss = sess.run(self.loss, feed_dict={self.input_data: batch_input_data})
				print >> sys.stderr, "[%s] Iter: %d" % (arrow.now(), (step * self.batch_size))
				print >> sys.stderr, "[%s] Train Loss: %.5f" % (arrow.now(), train_loss)
				# Plot intensity
				imit_times = sess.run(self.learner_times)
				intensityplot4seqs(imit_times, batch_input_data[:, :, 0], T=self.t_max, n_t=15.,
				                   file_path="resource/img/intensity/iter_%d.png" % step)
			step += 1
		print >> sys.stderr, "[%s] Optimization Finished!" % arrow.now()

	# Utils Function

	# def unittest(self, sess):
	# 	init = tf.global_variables_initializer()
	# 	sess.run(init)
    #
	# 	test_output = sess.run(self.loss, feed_dict={self.input_data: [
	# 		[[0.1], [0.5], [1.], [1.5], [1.7], [2]],
	# 		[[0.1], [0.5], [1.], [1.5], [1.7], [2]],
	# 		[[0.1], [0.5], [1.], [1.5], [1.7], [2]],
	# 		[[0.1], [0.5], [1.], [1.5], [1.7], [2]],
	# 		[[0.1], [0.5], [1.], [1.5], [1.7], [2]]
	# 	]})
	# 	print test_output

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

if __name__ == "__main__":
	# Configuration parameters
	seq_len      = 6
	batch_size   = 5
	state_size   = 3
	feature_size = 1  # Please fix feature_size to 1, since ppg only supports 1D feature for the time being
	t_max        = 3.
	data_size    = 6000
	generate_iters = 50
	training_iters = 20000

	ppg = PointProcessGenerator(
		t_max=t_max,
		seq_len=seq_len,
		batch_size=batch_size,
		state_size=state_size,
		feature_size=feature_size,
		iters=training_iters, display_step=10, lr=1e-5)

	with tf.Session() as sess:
		ppg.unittest(sess)
