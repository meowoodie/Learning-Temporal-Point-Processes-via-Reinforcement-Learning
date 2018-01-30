#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for visualizing results of experiments
"""

import sys
import arrow
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from ppgen import *

def get_intensity(seqs, n_t=50, t0=0, T=None):
    """
    Calculate intensity (pdf) of input sequences

    Optional method: calculating histogram for the sequences
    """
    seq     = np.array(seqs).flatten()
    T       = seq.max() if T is None else T
    delta_t = (T - t0) / n_t

    cdf = [ len(filter(lambda x: t < cdf_t, seq))
            for cdf_t in np.arange(t0, T+delta_t, delta_t) ]
    pdf = [ float(cur_cdf - prv_cdf)/float(len(seq))
            for prv_cdf, cur_cdf in zip(cdf[:-1], cdf[1:]) ]

    return pdf, np.arange(t0, T, delta_t)

def get_integral_diffs(seqs, intensity):
    integral_diffs = []
    for seq in seqs:
        seq_indice    = range(len(filter(lambda t: t>0, seq)))
        integral_diff = [ intensity.get_integral(seq[cur_ind], seq[:cur_ind]) - \
                          intensity.get_integral(seq[prv_ind], seq[:prv_ind])
                          for prv_ind, cur_ind in zip(seq_indice[:-1], seq_indice[1:]) ]
        integral_diff.insert(0, intensity.get_integral(seq[0], []) - 0)
        integral_diffs += integral_diff
    return integral_diffs

# Plotting Methods

def qqplot4seqs(time_seqs,
                x_left=0., x_right=8., y_left=0., y_right=8.,
                output_path="resource/img/qqplot/test"):
    """
    Plot Normal Q-Q plot for homogenuos poisson temporal sequences.

    Compare the quantiles between the distribution of differences of elements in
    input sequence with standard exponential distribution
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_left, y_right)
    # Calculate differences (delta values) between each two consecutive points
    # And concatenate the first elements in each sequence with their following delta values
    # Finally set last elements (negative) in each sequence to 0 (because last elements = 0 - last time < 0)
    delta_points = np.concatenate([
        time_seqs[:,[0]], np.diff(time_seqs, axis=1)], axis=1).clip(min=0).flatten()
    nonzero_delta_points = delta_points[np.nonzero(delta_points)]
    stats.probplot(nonzero_delta_points, dist=stats.expon, fit=True, plot=ax)
    plt.savefig(output_path)

def qqplot4intdiff(intdiff):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    stats.probplot(intdiff, dist=stats.expon, fit=True, plot=ax)
    plt.show()
