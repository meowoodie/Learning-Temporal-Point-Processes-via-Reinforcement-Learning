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

def get_intensity(seqs, T=None, n_t=50, t0=0):
    if T is None:
        T = np.array(seqs).flatten().max()

    dt = (T - t0) / n_t
    ts = np.arange(t0, T, dt)
    n_seqs = len(seqs)
    cnt    = np.zeros((n_t, 1))

    for i in range(n_seqs):
        seq = seqs[i]
        j   = 0
        k   = 0
        for t in np.arange(t0+dt, T+dt, dt):
            while (j < len(seq) and seq[j] <= t):
                j = j + 1
            cnt[k] = cnt[k] + j
            k = k + 1
    dif = np.zeros((len(cnt),1))
    dif[0] = cnt[0]
    for i in range(len(cnt)-1):
        dif[i+1] = cnt[i+1]-cnt[i]
    intensity = dif/(n_seqs)/dt

    return ts, intensity


def get_integral_empirical(sequences, intensity, T, n_t, t0=None):

    if T is None:
        T = max(max(sequences))
    if n_t is None:
        n_t = 50
    if t0 is None:
        t0 = 0
    dt = (T-t0)/n_t
    ts = np.arange(t0,T,dt)
    n_seqs = len(sequences)

    integral = []
    for i in range(1000):
        seq = sequences[i]
        integral_seq = []
        for j in range(len(seq)-1):
            t_start = seq[j]
            t_end = seq[j+1]
            index_start = np.int( t_start/dt)
            index_end = np.int(t_end/dt)+1
            integral_seq.append( np.sum(intensity[index_start:index_end])*dt -intensity[index_start]*(t_start-index_start*dt)-intensity[index_end-1]*(index_end*dt-t_end))
        integral += integral_seq
    return integral


def hawkes_integral(sequences,model):
    integrals = []
    for seq in sequences:
        integral = []
        seq = np.asarray(seq)
        for i in range(len(seq)-1):
            integral_delta = (seq[i+1]-seq[i])*model['mu'] + model['alpha'] * np.sum(np.exp(-(seq[i]-seq[:i+1]))-np.exp(-(seq[i+1]-seq[:i+1])))
            integral.append(integral_delta)
        integrals += integral
    return integrals

# def selfcorrecting_integral(sequences,model):
#     integrals = []
#     for seq in sequences:
#         integral = []
#         seq = np.asarray(seq)
#         for i in range(len(seq)-1):
#             integral_delta = (np.exp(model['mu']*seq[i+1]) - np.exp(model['mu']*seq[i]))/np.exp(model['alpha']*len(seq[:i+1]))/model['mu']
#             integral.append(integral_delta)
#         integrals += integral
#     return integrals

def gaussian_integral(sequences,model):
    integrals = []
    for seq in sequences:
        integral = []
        seq = np.asarray(seq)
        for i in range(len(seq)-1):
            integral_delta = np.sum( model['coef'] * (scipy.stats.norm.cdf(seq[i+1], model['center'], model['std']) - scipy.stats.norm.cdf(seq[i], model['center'], model['std']) ) )
            integral.append(integral_delta)
        integrals+=integral
    return integrals

def homo_poisson_qqplot(time_seqs, x_left=0., x_right=8., y_left=0., y_right=8.,
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
