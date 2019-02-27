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

from ppgrl import RL_Hawkes_Generator

# Avoid error msg [OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.]
# Reference: https://github.com/dmlc/xgboost/issues/1715
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
	# generate expert sequences
	# np.random.seed(0)
	# tf.set_random_seed(1)

	points = np.load('../Spatio-Temporal-Point-Process-Simulator/results/hpp_Feb_18.npy')
	print(points.shape)

	expert_seq_t = np.expand_dims(points[:, :, 0], -1)
	expert_seq_l = points[:, :, 1:]

	# training model
	with tf.Session() as sess:
		# model configuration
		batch_size       = 10
		epoches          = 10

		ppg = RL_Hawkes_Generator(T=[0., 10.], S=[[-1., 1.], [-1., 1.]], maximum=1e+3)

		ppg.train(sess,
			batch_size, epoches, 
			expert_seq_t, expert_seq_l,
			lr=1e-6,
			trainplot=False)
