#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import sys
import arrow
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def qqplot(time_seqs, x_left=0., x_right=8., y_left=0., y_right=8.,
           output_path="resource/img/qqplot/test"):
    """
    Plot Q-Q plot for time sequences.
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_left, y_right)
    # ax.plot([x_left, y_left], [x_right, y_right], color="b", linestyle='-', linewidth=2)
    # Calculate differences (delta values) between each two consecutive points
    # And concatenate the first elements in each sequence with their following delta values
    # Finally set last elements (negative) in each sequence to 0 (because last elements = 0 - last time < 0)
    # delta_points = np.diff(time_seqs, axis=1).clip(min=0)
    delta_points = np.concatenate([
        time_seqs[:,[0]], np.diff(time_seqs, axis=1)], axis=1).clip(min=0).flatten()
    nonzero_delta_points = delta_points[np.nonzero(delta_points)]
    stats.probplot(nonzero_delta_points, dist=stats.expon, fit=True, plot=ax)
    plt.savefig(output_path)

if __name__ == "__main__":
    test_file = "seql30.bts64.sts64.fts1.tmx15.dts2000.txt"
    test_seqs = np.loadtxt("resource/generation/%s" % test_file, delimiter=",")
    # print qqplot(test_seqs)
