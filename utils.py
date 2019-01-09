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
    x_sqr = tf.expand_dims(tf.reduce_sum(x * x, 1), -1) # [length, 1]
    y_sqr = tf.expand_dims(tf.reduce_sum(y * y, 1), -1) # [length, 1]
    xy    = tf.matmul(x, tf.transpose(y))               # [length, length]
    dist_mat = x_sqr + tf.transpose(y_sqr) - 2 * xy
    return dist_mat

class PointProcessIntensityMeter(object):

    def __init__(self, T, batch_size):
        self.batch_size = batch_size
        self.T          = T
        # figure and axes for time intensity plot
        self.fig_t      = plt.figure()
        self.ax_t       = self.fig_t.add_subplot(111)
        # figure and axes for space intensity plot
        self.fig_l      = plt.figure()
        self.ax_l1      = self.fig_l.add_subplot(1,2,1)
        self.ax_l2      = self.fig_l.add_subplot(1,2,2)
        plt.ion()

    def update_time_intensity(self, seq_t_1, seq_t_2, tlim=10):
        # clear last figure
        self.ax_t.clear()
        # sequence 1
        seq_flat_1 = seq_t_1.flatten()
        seq_flat_1 = seq_flat_1[seq_flat_1 != 0]
        seq_1_intensity_cum = []
        for grid in np.arange(0, self.T, 0.5):
            idx = (seq_flat_1 < grid)
            event_count_cum = len(seq_flat_1[idx])
            seq_1_intensity_cum = np.append(seq_1_intensity_cum, event_count_cum)
        seq_1_intensity = np.append(seq_1_intensity_cum[0], np.diff(seq_1_intensity_cum)) / self.batch_size
        self.ax_t.plot(np.arange(0, self.T, 0.5), seq_1_intensity)
        # sequence 2
        seq_flat_2 = seq_t_2.flatten()
        seq_2_intensity_cum = []
        for grid in np.arange(0, self.T, 0.5):
            idx = (seq_flat_2 < grid)
            event_count_cum = len(seq_flat_2[idx])
            seq_2_intensity_cum = np.append(seq_2_intensity_cum, event_count_cum)
        seq_2_intensity = np.append(seq_2_intensity_cum[0], np.diff(seq_2_intensity_cum)) / self.batch_size
        self.ax_t.plot(np.arange(0, self.T, 0.5), seq_2_intensity)
        # configure plot limits
        self.ax_t.set_ylim((0, tlim))
        plt.pause(0.02)

    def update_location_intensity(self, seq_l_1, seq_l_2, xylim=5, gridsize=51):
        # clear last figure
        self.ax_l1.clear()
        self.ax_l2.clear()
        # configure bins for histogram
        xedges = np.linspace(-xylim, xylim, gridsize)
        yedges = np.linspace(-xylim, xylim, gridsize)
        # sequence 1
        seq_1_x = seq_l_1[:, :, 0].flatten()
        seq_1_y = seq_l_1[:, :, 1].flatten()
        H, xedges, yedges = np.histogram2d(seq_1_x, seq_1_y, bins=(xedges, yedges))
        self.ax_l1.imshow(H.T, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        # sequence 2
        seq_2_x = seq_l_2[:, :, 0].flatten()
        seq_2_y = seq_l_2[:, :, 1].flatten()
        H, xedges, yedges = np.histogram2d(seq_2_x, seq_2_y, bins=(xedges, yedges))
        self.ax_l2.imshow(H.T, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        # configure plot limits
        self.ax_l1.set_xlim((-xylim, xylim))
        self.ax_l1.set_ylim((-xylim, xylim))
        self.ax_l2.set_xlim((-xylim, xylim))
        self.ax_l2.set_ylim((-xylim, xylim))
        plt.pause(0.02)