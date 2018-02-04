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
from scipy import stats
import matplotlib.pyplot as plt

from utils.plots import *

class PointProcessGenerator6(object):
	"""

	"""

	def __init__(self, kernel_bandwidth=1, T_max=10, state_size=1, seq_len=3, batch_size=2):

		self.T_max = T_max
		self.kernel_bandwidth	= kernel_bandwidth

		self.state_size = state_size
		self.seq_len = seq_len
		self.batch_size = batch_size

		self.knock_out = 1- tf.contrib.kfac.utils.kronecker_product(
			tf.eye(2*self.batch_size),
			tf.ones([self.seq_len, self.seq_len])
		)

		self.W_f = tf.Variable(0.0001 * tf.random_normal([state_size + 1, state_size]))
		self.b_f = tf.Variable(0.0001 * tf.random_normal([1, state_size]))

		self.W_i = tf.Variable(0.0001 * tf.random_normal([state_size + 1, state_size]))
		self.b_i = tf.Variable(0.0001 * tf.random_normal([1, state_size]))

		self.W_C = tf.Variable(0.0001 * tf.random_normal([state_size + 1, state_size]))
		self.b_C = tf.Variable(0.0001 * tf.random_normal([1, state_size]))

		self.W_o = tf.Variable(0.0001 * tf.random_normal([state_size + 1, state_size]))
		self.b_o = tf.Variable(0.0001 * tf.random_normal([1, state_size]))

		self.V = tf.Variable(0.0001 * tf.random_normal([state_size, 1]))
		self.b2 = tf.Variable(0.0001 * tf.random_normal([1]))

		# self.W = 0.01 * tf.random_normal([state_size + 1, state_size])
		# self.b1 = 0.01 * tf.random_normal([1, state_size])
		# self.V = 0.01 * tf.random_normal([state_size, 1])
		# self.b2 = 0.01 * tf.random_normal([1])
		# self.W = tf.constant(np.array([[0.2],[0.3]], dtype=np.float32))
		# self.b1 = tf.constant(np.array([[0.4]], dtype=np.float32))
		# self.V = tf.constant(np.array([[0.7]], dtype=np.float32))
		# self.b2 = tf.constant(np.array([0.6], dtype=np.float32))

		# self.W = tf.Variable(0.00001 * tf.random_normal([state_size + 1, state_size]))
		# self.W2 = tf.Variable(0.00001 * tf.random_normal([state_size, state_size]))
		# self.W3 = tf.Variable(0.00001 * tf.random_normal([state_size, state_size]))
		# self.b1 = tf.Variable(0.00001 * tf.random_normal([1, state_size]))
		# self.b3 = tf.Variable(0.00001 * tf.random_normal([1, state_size]))
		# self.V = tf.Variable(0.00001 * tf.random_normal([state_size, 1]))
		# self.b2 = tf.Variable(0.00001 * tf.random_normal([1]))
		# self.VV = tf.Variable(0.00001 * tf.random_normal([state_size, 1]))
		# self.bb1 = tf.Variable(0.00001 * tf.random_normal([1, state_size]))
		# self.bb2 = tf.Variable(0.00001 * tf.random_normal([1]))
		# self.bb3 = tf.Variable(0.00001 * tf.random_normal([1, state_size]))
        #
		# self.Wz = tf.Variable(0.00001 * tf.random_normal([state_size + 1, state_size]))
		# self.Wr = tf.Variable(0.00001 * tf.random_normal([state_size + 1, state_size]))
		# self.Ww = tf.Variable(0.00001 * tf.random_normal([state_size + 1, state_size]))

	def reward5(self, expert_time, expert_time_mask,
				learner_time2, learner_time_mask2, loglik_interval_mask):

		# first compute mmd2 and then do all the reinforce trick.

		# first compute the entire kernel matrix in one shot
		basis_time = tf.concat([expert_time, learner_time2], axis=0)
		norm2_kernel = tf.exp(
			-tf.square(basis_time - tf.transpose(basis_time)) / (self.kernel_bandwidth)
		)
		# knock out all self interaction terms to get rid of the bias
		norm2_kernel = tf.multiply(norm2_kernel, self.knock_out)

		block_size = self.batch_size*self.seq_len

		result = tf.zeros([1])

		# upper-left block, does not contain the policy we are optimizing.
		result += tf.matmul(
			expert_time_mask,
			tf.matmul(
				norm2_kernel[:block_size, :block_size],
				tf.transpose(expert_time_mask)
			)
		)

		# upper-right and lower-left block, contain one copy of the policy we are optimizing.
		result -= 2*tf.matmul(
			expert_time_mask,
			tf.matmul(
				norm2_kernel[:block_size, block_size:],
				tf.transpose(learner_time_mask2 * loglik_interval_mask)
			)
		)

		# lower-right block, contains two copies of the policy we are optimizing.
		# hence using the rule of total derivatives, we have 2 copies here.
		result += 2*tf.matmul(
			learner_time_mask2,
			tf.matmul(
				norm2_kernel[block_size:, block_size:],
				tf.transpose(learner_time_mask2 * loglik_interval_mask)
			)
		)

		print("reward v 6")

		# since sqrt is monotonic transformation, it can be removed in the optimization.
		return result / (self.batch_size * self.batch_size)

		# norm = tf.sqrt(norm2)
		# return norm

	def generate3(self, rand_uniform_pool):

		hidden_state = tf.zeros([self.batch_size, self.state_size], tf.float32)
		C_t = tf.zeros([self.batch_size, self.state_size], tf.float32)
		# output = tf.zeros([self.batch_size, 1], tf.float32)
		cum_output = tf.zeros([self.batch_size, 1], tf.float32)

		output_array = []
		cum_output_array = []
		for i in range(self.seq_len):
			# sigma = tf.exp(tf.matmul(hidden_state, self.V) + self.b2)
			sigma = tf.nn.elu(tf.matmul(hidden_state, self.V) + self.b2) + 1

			# the number 2 here is just make sure we are learning.
			output = (
				-2 * tf.log(
					tf.reshape(rand_uniform_pool[:, i], [self.batch_size,1]))) / sigma

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

			output_array.append(output)
			cum_output_array.append(cum_output)

		# print(output.shape)
		# print(cum_output.shape)
		print("latest version 6")

		return tf.concat(output_array, axis=1), tf.concat(cum_output_array, axis=1)

	def compute_loglik_interval(self, learner_time_interval2, learner_time_mat_mask2, final_time_mask2):

		hidden_state = tf.zeros([self.batch_size, self.state_size], tf.float32)
		C_t = tf.zeros([self.batch_size, self.state_size], tf.float32)

		cum_output = tf.zeros([self.batch_size, 1])
		loglik_interval_array = []
		log_survival_array = []
		for i in range(self.seq_len):
			# the number 2 here is to compensate the 2 we put in the generator.
			# sigma = tf.exp(tf.matmul(hidden_state, self.V) + self.b2) / 2
			sigma = (tf.nn.elu(tf.matmul(hidden_state, self.V) + self.b2) + 1) / 2.0

			output = tf.reshape(learner_time_interval2[:,i], [self.batch_size, 1])
			# output_mask = tf.reshape(learner_time_mat_mask2[:,i], [self.batch_size, 1])

			loglik_interval = (tf.log(sigma) - sigma * output)
			loglik_interval_array.append(loglik_interval)
			log_survival = -sigma * (self.T_max-cum_output)
			log_survival_array.append(log_survival)

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

		tmp_sum = tf.reduce_sum(
			learner_time_mat_mask2 * tf.concat(loglik_interval_array, axis=1),
			axis=1
		)
		# adding the survival term with mask, in general it seems that tf.Tensor object
		# works with masking but not direct indexing
		tmp_survival = tf.reduce_sum(
			final_time_mask2 * tf.concat(log_survival_array, axis=1),
			axis=1
		)
		tmp_sum += tmp_survival

		print("new comp loglik 6")

		# since entire sequence gets the same log-likelihood term,
		# we replicate it to the entire sequence
		return tf.reshape(tmp_sum, [self.batch_size, 1]) * tf.ones([1,self.seq_len])

	def _next_batch_index(self, num_seq, start_index):
		"""
		Next Batch

		"""

		start_seq = start_index % num_seq
		# end index of the row of the input data
		end_seq = (start_seq + self.batch_size) % num_seq
		# if there is not enough data left in the dataset for generating an integral batch,
		# then top up this batch by traversing back to the start of the dataset.
		if end_seq < start_seq:
			batch_index = range(start_seq, num_seq) + range(end_seq)
		else:
			batch_index = range(start_seq, end_seq)

		# Update the start index
		start_index += self.batch_size
		return batch_index, start_index

	def train3(self, sess, expert_time_pool, outer_iters=1000, inner_iters=5, display_step=5):

		expert_time = tf.placeholder(
			tf.float32,
			(self.batch_size*self.seq_len, 1)
		)
		expert_time_mask = tf.placeholder(
			tf.float32,
			(1, self.batch_size*self.seq_len)
		)
		rand_uniform_pool = tf.placeholder(
			tf.float32,
			(self.batch_size, self.seq_len)
		)

		learner_interval_mat2 = tf.placeholder(
			tf.float32,
			(self.batch_size, self.seq_len)
		)
		learner_time2 = tf.placeholder(
			tf.float32,
			(self.batch_size*self.seq_len, 1)
		)
		learner_time_mat_mask2 = tf.placeholder(
			tf.float32,
			(self.batch_size, self.seq_len)
		)
		learner_time_mask2 = tf.placeholder(
			tf.float32,
			(1, self.batch_size*self.seq_len)
		)
		final_time_mask2 = tf.placeholder(
			tf.float32,
			(self.batch_size, self.seq_len)
		)

		learner_interval_mat, learner_time_mat  = self.generate3(rand_uniform_pool)
		learner_time_mat_mask = tf.cast(learner_time_mat < self.T_max, tf.float32)
		learner_time = tf.reshape(learner_time_mat, [self.batch_size*self.seq_len, 1])
		learner_time_mask = tf.cast(learner_time < self.T_max, tf.float32)
		learner_time_mask = tf.reshape(learner_time_mask, [1, self.batch_size*self.seq_len])

		# simplified version where we have ignored survival term.
		loglik_interval_mask = self.compute_loglik_interval(
			learner_interval_mat2,
			learner_time_mat_mask2,
			final_time_mask2
		)
		loglik_interval_mask = tf.reshape(loglik_interval_mask, [1, self.batch_size*self.seq_len])

		seq_rewards = self.reward5(
			expert_time, expert_time_mask,
			learner_time2, learner_time_mask2, loglik_interval_mask
		)

		global_step = tf.Variable(0, trainable=False)
		starter_learning_rate = 0.01
		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
												   50, 0.96, staircase=True)
		optimizer = tf.train.AdamOptimizer(
			learning_rate, beta1 = 0.9, beta2 = 0.9
		).minimize(seq_rewards, global_step=global_step)

		# optimizer = tf.train.AdamOptimizer(
		# 	learning_rate=1e-3, beta1 = 0.9, beta2 = 0.9
		# ).minimize(seq_rewards)

		# optimizer = tf.train.AdagradOptimizer(
		# 	learning_rate = 1e-2
		# ).minimize(seq_rewards)

		# # Passing global_step to minimize() will increment it at each step.
		# optimizer = (
		# 	tf.train.GradientDescentOptimizer(learning_rate).minimize(seq_rewards, global_step=global_step)
		# )

		# step = 0
		start_index = 0

		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# plt.ion()

		# Add an op to initialize the variables.

		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		for step in range(outer_iters):

			# generate sequences according to current generative model
			batch_index, start_index = self._next_batch_index(expert_time_pool.shape[0], start_index)
			print(start_index)
			print(batch_index)
			cur_expert_time = np.reshape(
				expert_time_pool[batch_index,:],
				(self.batch_size*self.seq_len, 1)
			)
			cur_expert_time_mask = np.reshape(
				(cur_expert_time < self.T_max).astype(np.float32),
				(1, self.batch_size*self.seq_len)
			)

			cur_rand_uniform_pool = np.random.rand(self.batch_size, self.seq_len).astype(np.float32)

			for i in range(inner_iters):

				# cur_rand_uniform_pool = np.random.rand(self.batch_size, self.seq_len).astype(np.float32)

				learner_interval_mat3, learner_time_mat3, learner_time_mat_mask3, learner_time3, learner_time_mask3 = sess.run(
					[learner_interval_mat, learner_time_mat, learner_time_mat_mask, learner_time, learner_time_mask],
					feed_dict={
						expert_time: cur_expert_time,
						expert_time_mask: cur_expert_time_mask,
						rand_uniform_pool: cur_rand_uniform_pool
					}
				)

				final_time_idx3 = np.argmax(learner_time_mat3 * learner_time_mat_mask3, axis=1)
				final_time_mask3 = np.zeros((self.batch_size, self.seq_len))
				for j in range(self.batch_size):
					final_time_mask3[j,min([final_time_idx3[j]+1, self.seq_len-1])] = 1.0

				# optimize the parameters of the generator model
				sess.run(optimizer, feed_dict={
					expert_time: cur_expert_time,
					expert_time_mask: cur_expert_time_mask,
					learner_time2: learner_time3,
					learner_time_mask2: learner_time_mask3,
					learner_interval_mat2: learner_interval_mat3,
					learner_time_mat_mask2: learner_time_mat_mask3,
					final_time_mask2: final_time_mask3
				})

				if i % display_step == 0:
					# Calculate batch loss and accuracy
					test_point_mat, train_loss = sess.run(
						[learner_time_mat, seq_rewards], feed_dict={
							expert_time: cur_expert_time,
							expert_time_mask: cur_expert_time_mask,
							rand_uniform_pool: cur_rand_uniform_pool,
							learner_time2: learner_time3,
							learner_time_mask2: learner_time_mask3,
							learner_interval_mat2: learner_interval_mat3,
							learner_time_mat_mask2: learner_time_mat_mask3,
							final_time_mask2: final_time_mask3
						})

					print >> sys.stderr, "[%s] Outer: %d, Inner: %d, loss: %f" \
										 % (arrow.now(), step, i, train_loss)

					test_point_diff = np.concatenate(
						[np.reshape(test_point_mat[:,[0]], (self.batch_size,1)),
									np.diff(test_point_mat, axis=1)],
						axis=1
					)
					test_point_flat = test_point_mat.flatten()
					idx = (test_point_flat < self.T_max)

					# ax.clear()
					# res, res2 = stats.probplot(
					# 	test_point_diff.flatten()[idx],
					# 	dist=stats.expon, fit=True, plot=ax
					# )
					# # print(res[0])
					# plt.plot(res[0], res[0])

					train_point_diff = np.concatenate(
						[np.reshape(expert_time_pool[batch_index,[0]], (self.batch_size,1)),
						 np.diff(expert_time_pool[batch_index,:], axis=1)],
						axis=1
					)

					print("there")

					train_point_flat = expert_time_pool[batch_index,:].flatten()

					intensityplot4seqs(test_point_mat, expert_time_pool[batch_index,:],
					                   T=self.T_max, n_t=20, t0=0,
					                   file_path="resource/img/intensity/iter_%d.png" % step)
					# idx = (train_point_flat < self.T_max)
					# res, res2 = stats.probplot(
					# 	train_point_diff.flatten()[idx],
					# 	dist=stats.expon, fit=True, plot=ax
					# )
                    #
					# plt.pause(0.02)

		print("W:")
		print(sess.run(self.W))

		print("b1:")
		print(sess.run(self.b1))

		print("V:")
		print(sess.run(self.V))

		print("b2:")
		print(sess.run(self.b2))
