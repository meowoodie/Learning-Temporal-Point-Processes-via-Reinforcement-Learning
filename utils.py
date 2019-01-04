#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

class PointProcessIntensityMeter(object):

    def __init__(self, T, batch_size):
        self.batch_size = batch_size
        self.T          = T
        self.fig        = plt.figure()
        self.ax         = self.fig.add_subplot(111)
        plt.ion()

    def update_time_intensity(self, seq_t_1, seq_t_2):
        # clear last figure
        self.ax.clear()
        # sequence 1
        seq_flat_1 = seq_t_1.flatten()
        seq_1_intensity_cum = []
        for grid in np.arange(0, self.T, 0.5):
            idx = (seq_flat_1 < grid)
            event_count_cum = len(seq_flat_1[idx])
            seq_1_intensity_cum = np.append(seq_1_intensity_cum, event_count_cum)
        seq_1_intensity = np.append(seq_1_intensity_cum[0], np.diff(seq_1_intensity_cum)) / self.batch_size
        plt.plot(np.arange(0, self.T, 0.5), seq_1_intensity)
        # sequence 2
        seq_flat_2 = seq_t_2.flatten()
        seq_2_intensity_cum = []
        for grid in np.arange(0, self.T, 0.5):
            idx = (seq_flat_2 < grid)
            event_count_cum = len(seq_flat_2[idx])
            seq_2_intensity_cum = np.append(seq_2_intensity_cum, event_count_cum)
        seq_2_intensity = np.append(seq_2_intensity_cum[0], np.diff(seq_2_intensity_cum)) / self.batch_size
        plt.plot(np.arange(0, self.T, 0.5), seq_2_intensity)
        plt.ylim((0, 6))
        plt.pause(0.02)