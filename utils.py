#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def l2_norm(x, y):
    """
    This helper function calculates distance (l2 norm) between two arbitrary data points from tensor x and 
    tensor y respectively, where x and y have the same shape [length, data_dim].
    """
    x_sqr = tf.reduce_sum(x * x, 1)       # [length, 1]
    y_sqr = tf.reduce_sum(y * y, 1)       # [length, 1]
    xy    = tf.matmul(x, tf.transpose(y)) # [length, length]
    dist_mat = x_sqr + y_sqr - 2 * xy
    return dist_mat