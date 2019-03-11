#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

def lebesgue_measure(S):
    """
    A helper function for calculating the Lebesgue measure for a space.
    It actually is the length of an one-dimensional space, and the area of
    a two-dimensional space.
    """
    sub_lebesgue_ms = [ sub_space[1] - sub_space[0] for sub_space in S ]
    return np.prod(sub_lebesgue_ms)

def l2_norm(x, y):
    """
    This helper function calculates distance (l2 norm) between two arbitrary data points from tensor x and 
    tensor y respectively, where x and y have the same shape [length, data_dim].
    """
    x     = tf.cast(x, dtype=tf.float32)
    y     = tf.cast(y, dtype=tf.float32)
    x_sqr = tf.expand_dims(tf.reduce_sum(x * x, 1), -1) # [length, 1]
    y_sqr = tf.expand_dims(tf.reduce_sum(y * y, 1), -1) # [length, 1]
    xy    = tf.matmul(x, tf.transpose(y))               # [length, length]
    dist_mat = x_sqr + tf.transpose(y_sqr) - 2 * xy
    return dist_mat

class Meter(object):
    """
    Base class for the point process visualizer
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # figure and axes for time intensity plot
        self.fig_t      = plt.figure()
        self.ax_t       = self.fig_t.add_subplot(111)
        # figure and axes for space intensity plot
        self.fig_l      = plt.figure()
        self.ax_l1      = self.fig_l.add_subplot(1,2,1)
        self.ax_l2      = self.fig_l.add_subplot(1,2,2)
        plt.ion()
    
class PointProcessDistributionMeter(Meter):
    """
    Data distribution visualizer for point process
    """
    def __init__(self, T, S, batch_size):
        self.T = T
        self.S = S
        Meter.__init__(self, batch_size)

    def update_time_distribution(self, seq_t_learner, seq_t_expert):
        self.update_distribution(seq_t_learner, seq_t_expert, 
            self.ax_t, self.T, 
            xlabel="Time", ylabel="Distribution")

    def update_location_distribution(self, seq_l_learner, seq_l_expert):
        self.update_distribution(seq_l_learner[:, :, 0], seq_l_expert[:, :, 0], 
            self.ax_l1, self.S[0], 
            xlabel="X", ylabel="Distribution")
        self.update_distribution(seq_l_learner[:, :, 1], seq_l_expert[:, :, 1], 
            self.ax_l2, self.S[1], 
            xlabel="Y", ylabel="Distribution")

    @staticmethod
    def update_distribution(seq_learner, seq_expert, axes, xlim, xlabel, ylabel):
        # clear last figure
        axes.clear()
        seq_learner = seq_learner.flatten()
        seq_learner = seq_learner[seq_learner != 0]
        seq_expert  = seq_expert.flatten()
        seq_expert  = seq_expert[seq_expert != 0]
        sns.set(color_codes=True)
        sns.distplot(seq_learner, ax=axes, hist=False, rug=True, label="Learner")
        sns.distplot(seq_expert, ax=axes, hist=False, rug=True, label="Expert")
        axes.set_xlim(xlim)
        axes.set(xlabel=xlabel, ylabel=ylabel)
        axes.legend(frameon=False)
        plt.pause(0.02)
        
        

class PointProcessIntensityMeter(Meter):
    """
    Conditional intensity visualizer for point process
    """
    def __init__(self, T, batch_size):
        self.T = T
        Meter.__init__(self, batch_size)

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
        seq_flat_2 = seq_flat_2[seq_flat_2 != 0]
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



def plot_spatial_kernel(path, kernel, S, grid_size, 
        sigma_x_clim=[0.1, 0.35], sigma_y_clim=[0.1, 0.35], rho_clim=[-1, 1]):
    """
    Plot spatial kernel parameters over the spatial region, including 
    sigma_x, sigma_x, and rho. 
    """
    assert len(S) == 2, '%d is an invalid dimension of the space.' % len(S)
    # define the span for space region
    x_span = np.linspace(S[0][0], S[0][1], grid_size+1)[:-1]
    y_span = np.linspace(S[1][0], S[1][1], grid_size+1)[:-1]
    # map initialization
    sigma_x_map = np.zeros((grid_size, grid_size))
    sigma_y_map = np.zeros((grid_size, grid_size))
    rho_map     = np.zeros((grid_size, grid_size))
    # grid entris calculation
    for x_idx in range(grid_size):
        for y_idx in range(grid_size):
            s                         = [x_span[x_idx], y_span[y_idx]]
            sigma_x, sigma_y, rho     = kernel.nonlinear_mapping(s)
            sigma_x_map[x_idx][y_idx] = sigma_x
            sigma_y_map[x_idx][y_idx] = sigma_y
            rho_map[x_idx][y_idx]     = rho
    # plotting
    plt.rc('text', usetex=True)
    plt.rc("font", family="serif")
    # plot as a pdf file
    with PdfPages(path) as pdf:
        fig, axs = plt.subplots(1, 3)
        cmap     = matplotlib.cm.get_cmap('viridis')
        im_0     = axs[0].imshow(sigma_x_map, interpolation='nearest', origin='lower', cmap=cmap)
        im_1     = axs[1].imshow(sigma_y_map, interpolation='nearest', origin='lower', cmap=cmap)
        im_2     = axs[2].imshow(rho_map, interpolation='nearest', origin='lower', cmap=cmap)
        print(sigma_x_map.min(), sigma_x_map.max())
        print(sigma_y_map.min(), sigma_y_map.max())
        print(rho_map.min(), rho_map.max())
        # ticks for colorbars
        im_0.set_clim(*sigma_x_clim)
        im_1.set_clim(*sigma_y_clim)
        im_2.set_clim(*rho_clim)
        tick_0 = np.linspace(sigma_x_clim[0], sigma_x_clim[1], 5).tolist()
        tick_1 = np.linspace(sigma_y_clim[0], sigma_y_clim[1], 5).tolist()
        tick_2 = np.linspace(rho_clim[0], rho_clim[1], 5).tolist()
        # set x, y labels for subplots
        axs[0].set_xlabel(r'$x$')
        axs[1].set_xlabel(r'$x$')
        axs[2].set_xlabel(r'$x$')
        axs[0].set_ylabel(r'$y$')
        axs[1].set_ylabel(r'$y$')
        axs[2].set_ylabel(r'$y$')
        # remove x, y ticks
        axs[0].get_xaxis().set_ticks([])
        axs[1].get_xaxis().set_ticks([])
        axs[2].get_xaxis().set_ticks([])
        axs[0].get_yaxis().set_ticks([])
        axs[1].get_yaxis().set_ticks([])
        axs[2].get_yaxis().set_ticks([])
        # set subtitle for subplots
        axs[0].set_title(r'$\sigma_x$')
        axs[1].set_title(r'$\sigma_y$')
        axs[2].set_title(r'$\rho$')
        # plot colorbar
        cbar_0 = fig.colorbar(im_0, ax=axs[0], ticks=tick_0, fraction=0.046, pad=0.08, orientation="horizontal")
        cbar_1 = fig.colorbar(im_1, ax=axs[1], ticks=tick_1, fraction=0.046, pad=0.08, orientation="horizontal")
        cbar_2 = fig.colorbar(im_2, ax=axs[2], ticks=tick_2, fraction=0.046, pad=0.08, orientation="horizontal")
        # set font size of the ticks of the colorbars
        cbar_0.ax.tick_params(labelsize=5) 
        cbar_1.ax.tick_params(labelsize=5) 
        cbar_2.ax.tick_params(labelsize=5) 
        # adjust the width of the gap between subplots
        plt.subplots_adjust(wspace=0.2)
        pdf.savefig(fig)