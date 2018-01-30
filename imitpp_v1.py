#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kernel script for Imitation Learning for Point Process, mainly including class
PointProcessGenerator
"""

import sys
import arrow
import random
import numpy as np
import tensorflow as tf

# from utils.plots import qqplot

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

		# tf.set_random_seed(100)
		self.seq_len      = seq_len
		self.batch_size   = batch_size
		self.feature_size = feature_size
		self.state_size   = state_size
		self.iters        = iters
		self.display_step = display_step

		# create a BasicRNNCell
		self.rnn_cell = tf.contrib.rnn.BasicRNNCell(state_size)

		# input_data has shape [batch_size, sequence_len, feature_size]
		self.input_data = tf.placeholder(tf.float32, shape=[batch_size, seq_len, feature_size])
		self.t_max      = tf.constant(t_max, dtype=tf.float32)

		# Optimizer Computational Graph
		# expert actions and corresponding learner actions
		# expert_actions has shape [batch_size, seq_len, feature_size]
		# the orginal learner_actions has shape [seq_len, batch_size, feature_size]
		expert_actions          = self.input_data
		learner_actions, states = self._fixed_length_rnn(self.batch_size, rnn_len=self.seq_len)
		# learner_actions has shape [batch_size, seq_len, feature_size]
		# which now has the same shape as expert_actions
		learner_actions         = tf.transpose(learner_actions, perm=(1, 0, 2))

		# loss function
		self.loss = self._reward_loss(expert_actions, learner_actions)
		# optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9)\
		                   .minimize(self.loss)
		# self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)\
		#                    .minimize(self.loss)

		# Generator Computational Graph
		self.expert_times  = expert_actions[:, :, 0]
		self.learner_times = tf.multiply(tf.cast(learner_actions[:, :, 0] < self.t_max, tf.float32), learner_actions[:, :, 0])
		self.states        = states

	# Building Blocks of Computational Graph

	def _reward_loss(self, expert_actions, learner_actions):
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
		unfolded_learner_times = tf.multiply(unfolded_learner_times, learner_times_mask)
		# Concatenate expert_time and learner_time and prepare their mask for fast matrix operation
		basis_times      = tf.concat([unfolded_expert_times, unfolded_learner_times], axis=0)
		basis_times_mask = tf.transpose(tf.concat([expert_times_mask, -1 * learner_times_mask], axis=0))
		# Calculate the kernel bandwidth, which is the median value of expert_time and learner_time
		kernel_bandwidth = self._median_pairwise_distance(unfolded_expert_times, unfolded_learner_times)

		norm2_kernel = tf.exp(-tf.square(basis_times - tf.transpose(basis_times)) / 0.5) # kernel_bandwidth)
		# norm2_kernel = tf.subtract(norm2_kernel, tf.diag(tf.diag_part(norm2_kernel)))
		norm2 = tf.matmul(basis_times_mask,
		                  tf.matmul(norm2_kernel, tf.transpose(basis_times_mask))) / \
		        (self.batch_size * self.batch_size)
		mmd   = tf.sqrt(norm2)
		return mmd

	def _fixed_length_rnn(self, num_seq, rnn_len):
		"""
		Fixed Length RNN implemented by dynamic_rnn of tensorflow
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
		W = tf.get_variable("W_r", [self.state_size, self.feature_size])
		b = tf.get_variable("b_r", [self.feature_size], initializer=tf.constant_initializer(0.0))
		sigma = tf.nn.elu(tf.matmul(cur_state, W) + b) + 1
		stoch_action = sigma * 0.5 * -1 * tf.log(tf.random_uniform(tf.shape(sigma), minval=0, maxval=1))
		cur_action = tf.add(prv_action, stoch_action)
		# action has shape [num_seq, feature_size]
		# state has shape  [num_seq, state_size]
		return cur_action, cur_state

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
				# Plots
				# imit_times = sess.run(self.expert_times, feed_dict={self.input_data: batch_input_data})
				# qqplot(imit_times, output_path=("resource/img/qqplot/test"+str(step)))
				# Calculate batch loss and accuracy
				train_loss = sess.run(self.loss, feed_dict={self.input_data: batch_input_data})
				print >> sys.stderr, "[%s] Iter: %d" % (arrow.now(), (step * self.batch_size))
				print >> sys.stderr, "[%s] Train Loss: %.5f" % (arrow.now(), train_loss)
				if test_data is not None:
					test_loss  = sess.run(self.loss, feed_dict={self.input_data: test_data})
					print >> sys.stderr, "[%s] Test Loss: %.5f" % (arrow.now(), test_loss)
			step += 1
		print >> sys.stderr, "[%s] Optimization Finished!" % arrow.now()

	# Utils Function

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
