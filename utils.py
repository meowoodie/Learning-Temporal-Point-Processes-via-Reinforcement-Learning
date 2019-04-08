#!/usr/bin/env python
# -*- coding: utf-8 -*-
import branca
import folium
import geopandas
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon

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



class DataAdapter():
    """
    A helper class for normalizing and restoring data to the specific data range.
    
    init_data: numpy data points with shape [batch_size, seq_len, 3] that defines the x, y, t limits
    S:         data spatial range. eg. [[-1., 1.], [-1., 1.]]
    T:         data temporal range.  eg. [0., 10.]
    """
    def __init__(self, init_data, S=[[-1, 1], [-1, 1]], T=[0., 10.]):
        self.data = init_data
        self.T    = T
        self.S    = S
        self.tlim = [ init_data[:, :, 0].min(), init_data[:, :, 0].max() ]
        mask      = np.nonzero(init_data[:, :, 0])
        x_nonzero = init_data[:, :, 1][mask]
        y_nonzero = init_data[:, :, 2][mask]
        self.xlim = [ x_nonzero.min(), x_nonzero.max() ]
        self.ylim = [ y_nonzero.min(), y_nonzero.max() ]
        print(self.tlim)
        print(self.xlim)
        print(self.ylim)

    def normalize(self, data):
        """normalize batches of data points to the specified range"""
        rdata = np.copy(data)
        for b in range(len(rdata)):
            # scale x
            rdata[b, np.nonzero(rdata[b, :, 0]), 1] = \
                (rdata[b, np.nonzero(rdata[b, :, 0]), 1] - self.xlim[0]) / \
                (self.xlim[1] - self.xlim[0]) * (self.S[0][1] - self.S[0][0]) + self.S[0][0]
            # scale y
            rdata[b, np.nonzero(rdata[b, :, 0]), 2] = \
                (rdata[b, np.nonzero(rdata[b, :, 0]), 2] - self.ylim[0]) / \
                (self.ylim[1] - self.ylim[0]) * (self.S[1][1] - self.S[1][0]) + self.S[1][0]
            # scale t 
            rdata[b, np.nonzero(rdata[b, :, 0]), 0] = \
                (rdata[b, np.nonzero(rdata[b, :, 0]), 0] - self.tlim[0]) / \
                (self.tlim[1] - self.tlim[0]) * (self.T[1] - self.T[0]) + self.T[0]
        return rdata

    def restore(self, data):
        """restore the normalized batches of data points back to their real ranges."""
        ndata = np.copy(data)
        for b in range(len(ndata)):
            # scale x
            ndata[b, np.nonzero(ndata[b, :, 0]), 1] = \
                (ndata[b, np.nonzero(ndata[b, :, 0]), 1] - self.S[0][0]) / \
                (self.S[0][1] - self.S[0][0]) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
            # scale y
            ndata[b, np.nonzero(ndata[b, :, 0]), 2] = \
                (ndata[b, np.nonzero(ndata[b, :, 0]), 2] - self.S[1][0]) / \
                (self.S[1][1] - self.S[1][0]) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
            # scale t 
            ndata[b, np.nonzero(ndata[b, :, 0]), 0] = \
                (ndata[b, np.nonzero(ndata[b, :, 0]), 0] - self.T[0]) / \
                (self.T[1] - self.T[0]) * (self.tlim[1] - self.tlim[0]) + self.tlim[0]
        return ndata

    def normalize_location(self, x, y):
        """normalize a single data location to the specified range"""
        _x = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.S[0][1] - self.S[0][0]) + self.S[0][0]
        _y = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.S[1][1] - self.S[1][0]) + self.S[1][0]
        return np.array([_x, _y])

    def restore_location(self, x, y):
        """restore a single data location back to the its original range"""
        _x = (x - self.S[0][0]) / (self.S[0][1] - self.S[0][0]) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        _y = (y - self.S[1][0]) / (self.S[1][1] - self.S[1][0]) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return np.array([_x, _y])
    
    def __str__(self):
        raw_data_str = "raw data example:\n%s\n" % self.data[:1]
        nor_data_str = "normalized data example:\n%s" % self.normalize(self.data[:1])
        return raw_data_str + nor_data_str



def spatial_intensity_on_map(
    path, # html saving path
    da,   # data adapter object defined in utils.py
    lam,  # lambda object defined in stppg.py
    data, # a sequence of data points [seq_len, 3] happened in the past
    t,    # observation moment (t)
    xlim, # observation x range
    ylim, # observation y range
    ngrid=100):
    """Plot spatial intensity at time t over the entire map given its coordinates limits."""
    # data preparation
    # - remove the first element in the seq, since t_0 is always 0, 
    #   which will cause numerical issue when computing lambda value
    seqs = da.normalize(data)[:, 1:, :] 
    seq  = seqs[0]                          # only use the first seq
    seq  = seq[np.nonzero(seq[:, 0])[0], :] # only retain nonzero values
    seq_t, seq_s = seq[:, 0], seq[:, 1:] 
    sub_seq_t = seq_t[seq_t < t]            # only retain values before time t.
    sub_seq_s = seq_s[:len(sub_seq_t)]
    # generate spatial grid polygons
    xmin, xmax, width       = xlim[0], xlim[1], xlim[1] - xlim[0] 
    ymin, ymax, height      = ylim[0], ylim[1], ylim[1] - ylim[0]
    grid_height, grid_width = height / ngrid, width / ngrid
    x_left_origin   = xmin
    x_right_origin  = xmin + grid_width
    y_top_origin    = ymax
    y_bottom_origin = ymax - grid_height
    polygons        = [] # spatial polygons
    lam_dict        = {} # spatial intensity
    _id             = 0
    for i in range(ngrid):
        y_top    = y_top_origin
        y_bottom = y_bottom_origin
        for j in range(ngrid):
            # append the intensity value to the list
            s = da.normalize_location((x_left_origin + x_right_origin) / 2., (y_top + y_bottom) / 2.)
            v = lam.value(t, sub_seq_t, s, sub_seq_s)
            lam_dict[str(_id)] = 0 if np.isnan(v) else v
            _id += 1
            # append polygon to the list
            polygons.append(Polygon(
                [(y_top, x_left_origin), (y_top, x_right_origin), (y_bottom, x_right_origin), (y_bottom, x_left_origin)])) 
            # update coordinates
            y_top    = y_top - grid_height
            y_bottom = y_bottom - grid_height
        x_left_origin  += grid_width
        x_right_origin += grid_width
    # convert polygons to geopandas object
    geo_df = geopandas.GeoSeries(polygons) 
    # init map
    _map   = folium.Map(location=[sum(xlim)/2., sum(ylim)/2.], zoom_start=12, zoom_control=True)
    # plot polygons on the map
    lam_cm = branca.colormap.linear.YlOrRd_09.scale(min(lam_dict.values()), max(lam_dict.values())) # colorbar
    poi_cm = branca.colormap.linear.PuBu_09.scale(min(sub_seq_t), max(sub_seq_t)) # colorbar
    folium.GeoJson(
        data = geo_df.to_json(),
        style_function = lambda feature: {
            'fillColor':   lam_cm(lam_dict[feature['id']]),
            'fillOpacity': .5,
            'weight':      0.}).add_to(_map)
    # plot markers on the map
    for i in range(len(sub_seq_t)):
        x, y = da.restore_location(*sub_seq_s[i])
        folium.Circle(
            location=[x, y],
            radius=10, # sub_seq_t[i] * 100,
            color=poi_cm(sub_seq_t[i]),
            fill=True,
            fill_color='blue').add_to(_map)
    # save the map
    _map.save(path)