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
	batch_size   = 2
	state_size   = 4
	feature_size = 1
	input_data   = np.array([
		[[1], [1.5], [2], [0], [0]], [[1.1], [1.2], [2.4], [3.3], [5]], [[2], [2.2], [2.5], [3], [0]], [[1.1], [1.2], [2.4], [3.3], [5]], 
		[[1], [1.5], [2], [0], [0]], [[1.1], [1.2], [2.4], [3.3], [5]], [[2], [2.2], [2.5], [3], [0]], [[1.1], [1.2], [2.4], [3.3], [5]]])
	test_data    = np.array([
		[[1], [1.5], [2], [0], [0]], [[1.1], [1.2], [2.4], [3.3], [5]]])

	with tf.Session() as sess:

		ppg = PointProcessGenerater(
			seq_len=seq_len,
			batch_size=batch_size, 
			state_size=state_size,
			feature_size=feature_size)

		ppg.train(sess, input_data, test_data)
		actions, states_history = ppg.generate(sess, num_seq=3, max_t=7, pretrained=False)



