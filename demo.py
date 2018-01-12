#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import sys
import arrow
import random
import numpy as np
import tensorflow as tf
from imitpp import PointProcessGenerater

if __name__ == "__main__":
	#TODO: give more details and more examples on how to use it

	seq_len      = 5
	batch_size   = 3
	state_size   = 4
	feature_size = 1
	input_data   = [[[1], [1.5], [2], [0], [0]], [[1.1], [1.2], [2.4], [3.3], [5]], [[2], [2.2], [2.5], [3], [0]]]

	with tf.Session() as sess:

		ppg = PointProcessGenerater(
			seq_len=seq_len,
			batch_size=batch_size, 
			state_size=state_size,
			feature_size=feature_size)

		print ppg.train(sess, input_data)
		# print ppg.generate(sess, num_seq=3, max_t=7, max_learner_len=10)

		# A = tf.constant([[ 1.10000002], [ 1.20000005], [ 2.4000001 ], [ 3.29999995], [ 5.        ], [ 1.12912512], [ 1.60563707], [ 2.87550449], [ 3.63043833], [ 4.48129034]])
		# r = tf.reduce_sum(A*A, 1)

		# # turn r into column vector
		# r = tf.reshape(r, [-1, 1])
		# D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

		# a = tf.constant([1,2,3])
		# D = tf.transpose(tf.reshape(tf.tile(a, [2]), (2,3)))
		# sess = tf.Session()
		# print sess.run(D)

