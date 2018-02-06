#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment Synthetic Data
"""

import sys
import arrow
import random
import numpy as np
import tensorflow as tf

from utils.ppgen import IntensityHawkesPlusGaussianMixture, IntensityHomogenuosPoisson, generate_sample
from utils.plots import intensityplot4seqs
from imitpp_v1 import PointProcessGenerator

if __name__ == "__main__":
	# np.random.seed(100)

	# Configuration parameters
	# seq_len      = 30
	batch_size   = 32
	state_size   = 32
	feature_size = 1  # Please fix feature_size to 1, since ppg only supports 1D feature for the time being
	t_max        = 15.
	data_size    = 2000
	generate_iters = 50
	training_iters = 10000

	# Generate point process with complex intensity
	intensity = IntensityHawkesPlusGaussianMixture(mu=1, alpha=0.3, beta=1,
                                                   k=2, centers=[t_max/4., t_max*3./4.],
												   stds=[1, 1], coefs=[1, 1])
	# intensity = IntensityHomogenuosPoisson(lam=1.)
	ppsample  = generate_sample(intensity, T=t_max, n=data_size)
	seq_len   = max([ len(ppseq) for ppseq in ppsample ])
	# Check if max length of the poisson process sequences is less than the preset sequence length
	# if seq_len < max_len:
	# 	raise Exception("Insecure seq_len %d < max_len %d." % (seq_len, max_len))
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
			iters=training_iters, display_step=10, lr=1e-4)

		# # Loading well-trained model
		# file_name = "seql60.bts128.sts64.fts1.tmx15.dts6000"
		# tf_saver = tf.train.Saver()
		# tf_saver.restore(sess, "resource/model/%s" % file_name)

		# # Generating sequences
		# learner_actions = np.zeros((0, seq_len))
		# for ind in range(generate_iters):
		# 	seqs, _ = ppg.generate(sess, pretrained=True)
		# 	learner_actions = np.concatenate((learner_actions, seqs), axis=0)
        #
		# intensityplot4seqs(learner_actions, expert_actions, T=t_max, n_t=100., t0=0.,
		#                    file_path="results/intensityplot4seqs.png")

		# Training new model
		ppg.train(sess, expert_actions)
		tf_saver = tf.train.Saver()
		tf_saver.save(sess, "resource/model/seql%d.bts%d.sts%d.fts%d.tmx%d.dts%d" % \
		                    (seq_len, batch_size, state_size, feature_size, t_max, data_size))
