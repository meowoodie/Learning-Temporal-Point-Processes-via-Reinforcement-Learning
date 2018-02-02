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

def get_intensity(seq, n_seqs, n_t=100, t0=0, T=None):
    """
    Calculate intensity (pdf) of input sequences

    Optional method: calculating histogram for the sequences
    """
    T       = seq.max() if T is None else T
    delta_t = float(T - t0) / float(n_t)

    cdf = [ len(filter(lambda t: 0 < t and t < cdf_t, seq))
            for cdf_t in np.arange(t0, T+delta_t, delta_t) ]
    pdf = [ float(cur_cdf - prv_cdf) / float(n_seqs)
            for prv_cdf, cur_cdf in zip(cdf[:-1], cdf[1:]) ]

    return pdf, np.arange(t0, T, delta_t)

def get_integral_diffs(seqs, intensity, T_max):
    integral_diffs = []
    for seq in seqs:
        seq_indice    = range(len(filter(lambda t: t>0 and t<T_max, seq)))
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

def qqplot4intdiff(learner_seqs, expert_seqs, intensity, T,
                   file_path="results/qqplot4intdiff.png"):
    """
    Q-Q Plot for integral
    """
    # Init figure
    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111)
    # Get integral differences of intensity of sequences
    learner_intdiff = get_integral_diffs(learner_seqs, intensity, T_max)
    expert_intdiff  = get_integral_diffs(expert_seqs, intensity, T_max)
    # Plot Q-Q plot
    losm, losr = stats.probplot(learner_intdiff, dist=stats.expon, fit=True, plot=ax)
    eosm, eosr = stats.probplot(expert_intdiff, dist=stats.expon, fit=True, plot=ax)
    # Get handles of lines
    line_learner = ax.get_lines()[0]
    line_expert  = ax.get_lines()[2]
    line_learner_approx = ax.get_lines()[1]
    line_expert_approx  = ax.get_lines()[3]
    # Set color and legend
    line_learner.set_markerfacecolor('r')
    line_learner.set_markeredgecolor('r')
    line_learner_approx.set_color('r')
    line_learner_approx.set_color('r')
    line_expert.set_markerfacecolor('b')
    line_expert.set_markeredgecolor('b')
    line_expert_approx.set_color('b')
    line_expert_approx.set_color('b')
    line_learner.set_label("Learner Sequences")
    line_expert.set_label("Expert Sequences")
    # Plot groundtruth
    ax.plot(losm[0], losm[0], c='g')
    ax.legend()
    fig.savefig(file_path, bbox_inches='tight')

def intensityplot4seqs(learner_seqs, expert_seqs, T, n_t=100, t0=0,
                       file_path="results/intensityplot4seqs.png"):
    """
    Intensity Plot for Raw Sequences
    """
    # Flatten seqs
    len_expert_seqs  = len(expert_seqs) # nonhomogeneous length
    len_learner_seqs = len(learner_seqs)
    expert_seqs  = np.array(expert_seqs).flatten() # [item for sublist in expert_seqs for item in sublist]
    # learner_seqs = [item for sublist in learner_seqs for item in sublist]
    learner_seqs = np.array(learner_seqs).flatten()
    # Calculate intensity for expert and learner sequences
    expert_Y, expert_x   = get_intensity(expert_seqs, n_seqs=len_expert_seqs,
                                         n_t=n_t, t0=t0, T=T)
    learner_Y, learner_x = get_intensity(learner_seqs, n_seqs=len_learner_seqs,
                                         n_t=n_t, t0=t0, T=T)
    # Init figure and plot
    fig = plt.figure(figsize=(10, 10))
    plt.plot(expert_x, expert_Y)
    plt.plot(learner_x, learner_Y)
    plt.legend(["Expert", "Learner"], loc="lower right")
    plt.title("Intensity")
    fig.savefig(file_path, bbox_inches="tight")
    print "saving plots"



if __name__ == "__main__":

    T_max       = 15.

    print >> sys.stderr, "[%s] Loading learner sequences..." % arrow.now()

    intensity   = IntensityHawkesPlusGaussianMixture(mu=1, alpha=0.3, beta=1,
                                                     k=2, centers=[T_max/4., T_max*3./4.], stds=[1, 1], coefs=[1, 1])
    learner_seqs = np.loadtxt("resource/generation/hawkes_gaussianmixture_learner_seq.txt", delimiter=",")

    # intensity    = IntensityPoly(mu=1, alpha=0.3, beta=1,
    #                              segs=[0, T_max/4, T_max*2/4, T_max*3/4, T_max],
    #                              b=0, A=[1., -1., 1., -1.])
    # learner_seqs = np.loadtxt("resource/generation/hawkes_poly_learner_seq.txt", delimiter=",")

    print >> sys.stderr, "[%s] Generating expert sequences..." % arrow.now()
    expert_seqs = generate_sample(intensity, T=T_max, n=2000)

    # Plot 1: Q-Q plot
    print >> sys.stderr, "[%s] Plotting Q-Q plot..." % arrow.now()
    # qqplot4intdiff(learner_seqs, expert_seqs, intensity, T=T_max, file_path="results/qqplot4intdiff.png")

    # Plot 2: Intensity plot
    print >> sys.stderr, "[%s] Plotting Intensity plot..." % arrow.now()
    intensityplot4seqs(learner_seqs, expert_seqs, T=T_max, file_path="results/intensityplot4seqs.png")
