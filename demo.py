#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for testing basic version of imitpp on synthetic dataset
"""

import sys
import arrow
import random
import numpy as np
import tensorflow as tf

from mstppg import MSTPPGenerator
from imitpp import PointProcessGenerator

if __name__ == "__main__":
	# generate expert sequences
	np.random.seed(0)

	# data configuration
	n_seq       = 10
	T           = 10.
	m_dim       = 5

	# generating parameters
	alpha       = 0.6
	beta        = 1
	mu          = 2
	freq        = 1
	magn        = 1
	shift       = 0.5
	n_component = 3
	xlim        = [-5, 5]
	ylim        = [-5, 5]
	grid_time   = 0.1
	grid_space  = 1

	generator = MSTPPGenerator(
		num_seq=n_seq, T=T, mark_vol=m_dim,
		alpha=alpha, beta=beta, mu=mu, frequence=freq,
		magnitude=magn, shift=shift, num_component=n_component,
		xlim=xlim, ylim=ylim,
		grid_time=grid_time, grid_space=grid_space)

	_, expert_seq_t, expert_seq_l, expert_seq_m = generator.MSTPPSamples()
	print(expert_seq_t)
	print(expert_seq_t.shape)
	print(expert_seq_l.shape)
	print(expert_seq_m.shape)

	# training model
	tf.set_random_seed(1)
	with tf.Session() as sess:
		# model configuration
		step_size        = expert_seq_t.shape[1]
		lstm_hidden_size = 10
		loc_hidden_size  = 10
		mak_hidden_size  = 10
		batch_size       = 2
		epoches          = 10

		ppg = PointProcessGenerator(
			T=T, seq_len=step_size, 
			lstm_hidden_size=lstm_hidden_size, loc_hidden_size=loc_hidden_size, mak_hidden_size=mak_hidden_size, 
			m_dim=m_dim)

		ppg.train(sess, 
			batch_size, epoches, 
			expert_seq_t, expert_seq_l, expert_seq_m,
			train_test_ratio = 2.)
