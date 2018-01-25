#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple demo for generating poisson process sequences and use them to train a
poinsson process generator.
"""

import sys
import arrow
import random
import numpy as np
import tensorflow as tf
import Simulate_Poisson as SP
from imitpp import PointProcessGenerator

if __name__ == "__main__":
	# np.random.seed(103)

	# Configuration parameters
	seq_len      = 30
	batch_size   = 64
	state_size   = 64
	feature_size = 1  # Please fix feature_size to 1, since ppg only supports 1D feature for the time being
	t_max        = 15
	data_size    = 2000

	# Generate poisson process sequences
	pp       = SP.IntensityHomogenuosPoisson(1.0)
	ppsample = SP.generate_sample(pp, t_max, data_size)
	max_len  = max([ len(ppseq) for ppseq in ppsample ])
	# Check if max length of the poisson process sequences is less than the preset sequence length
	if seq_len < max_len:
		raise Exception("Insecure seq_len %d < max_len %d." % (seq_len, max_len))
	# Padding zeros for poisson process sequences
	expert_actions = np.zeros((data_size, seq_len, feature_size))
	for data_ind in range(data_size):
		for action_ind in range(len(ppsample[data_ind])):
			ppvalue = ppsample[data_ind][action_ind]
			expert_actions[data_ind, action_ind, 0] = ppvalue

	# Train point process generator by generated expert actions (poisson process sequences)
	with tf.Session() as sess:
		ppg = PointProcessGenerator(
			t_max=t_max,
			seq_len=seq_len,
			batch_size=batch_size,
			state_size=state_size,
			feature_size=feature_size,
			iters=10000, display_step=1, lr=1e-4)

		# Loading well-trained model
		# file_name = ...
		# tf_saver = tf.train.Saver()
        # tf_saver.restore(sess, "resource/model/%s" % file_name)

		# Training new model
		ppg.train(sess, expert_actions)
		tf_saver = tf.train.Saver()
		tf_saver.save(sess, "resource/model/seql%d.bts%d.sts%d.fts%d.tmx%d.dts%d" % \
		                    (seq_len, batch_size, state_size, feature_size, t_max, data_size))

		# # Generating
		# actions, states_history = ppg.generate(sess, pretrained=True)
		# np.savetxt("resource/generation/seql%d.bts%d.sts%d.fts%d.tmx%d.dts%d.txt" % \
		#            (seq_len, batch_size, state_size, feature_size, t_max, data_size), \
		# 		   actions, delimiter=',')
