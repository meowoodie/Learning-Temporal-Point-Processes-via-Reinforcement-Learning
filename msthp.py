#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Imitation Learning for Point Process

A Hawkes processes based model for generating marked spatial-temporal points.

References:
- https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator
- https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf   

Dependencies:
- Python 3.6.7
- tensorflow==1.5.0
"""

import utils
import numpy as np
import tensorflow as tf

class DiffusionKernel(object):
    '''
    Standard diffusion kernel function including the diffusion-type model proposed by Musmeci and
    Vere-Jones (1992).
    '''
    def __init__(self, beta=1., C=1., sigma=[1., 1.], offset=[0, 0]):
        self.beta   = tf.get_variable(name="kernel_beta", initializer=tf.constant(beta))
        self.sigma  = tf.get_variable(name="kernel_sigma", initializer=tf.constant(beta))
        self.offset = np.array(offset)
        self.C      = C

    def nu(self, delta_t, delta_l):
        return (self.C/(2*np.pi*tf.reduce_prod(self.sigma)*delta_t)) * \
               tf.exp(-1*self.beta*delta_t - tf.reduce_sum((tf.math.square(delta_l - self.offset) / tf.math.square(self.sigma)), axis=1) / (2*delta_t))

class MarkedSpatialTemporalHawkesProcess(object):
    """
    Marked Spatial Temporal Hawkes Process

    A stochastic marked spatial temporal points generator based on Hawkes process.
    """

    def __init__(self, kernel, mu=0, alpha=1., beta=1., max_lam=500., m_dim=None):
        """
        Params:
        - T:     maximum time
        - m_dim: number of categories of marks
        """
        self.mu      = tf.get_variable(name="lam_mu", initializer=tf.constant(mu))
        self.alpha   = tf.get_variable(name="lam_alpha", initializer=tf.constant(alpha))
        self.beta    = tf.get_variable(name="lam_beta", initializer=tf.constant(beta))
        self.max_lam = max_lam
        self.kernel  = kernel

    def _lam_value(self, seq_t, seq_s):
        '''
        return the intensity value at (t, s).
        The last element of seq_t and seq_s is the location (t, s) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        '''
        if tf.shape(seq_t)[0] > 1:
            # get current time, spatial values and historical time, spatial values.
            cur_t, his_t = seq_t[-1], seq_t[:-1]
            cur_s, his_s = seq_s[-1], seq_s[:-1]
            val = self.mu + self.alpha * tf.reduce_sum(self.beta * self.kernel.nu(cur_t-his_t, cur_s-his_s))
        else:
            val = self.mu
        return val

    def _lam_upper_bound(self):
        '''return the upper bound of the intensity value'''
        return self.max_lam

    # def _homogeneous_samples(self, T, x_lim, y_lim, batch_size):
    #     '''
    #     To generate a homogeneous Poisson point pattern in space S, this function
    #     uses a two step procedure:
    #     1. Simulate the number of events n = N(S) occurring in S according to a
    #     Poisson distribution with mean lam * |S|.
    #     2. Sample each of the n location according to a uniform distribution on S
    #     respectively.
    #     Returns samples: point process samples  
    #         [(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)]
    #     '''
    #     # sample space
    #     S  = [[0, T], x_lim, y_lim]
    #     # calculate the number of events in space S
    #     n  = utils.lebesgue_measure(S)
    #     Ns = tf.random_poisson([batch_size], lam=self._lam_upper_bound() * n, dtype=tf.int32)
    #     # simulate spatial sequence and temporal sequence separately.
    #     seq_ts = tf.random_uniform([batch_size, tf.reduce_max(Ns)], minval=0, maxval=T, dtype=tf.float32)
    #     seq_xs = tf.random_uniform([batch_size, tf.reduce_max(Ns)], minval=x_lim[0], maxval=x_lim[1], dtype=tf.float32)
    #     seq_ys = tf.random_uniform([batch_size, tf.reduce_max(Ns)], minval=y_lim[0], maxval=y_lim[1], dtype=tf.float32)
    #     seq_ls = tf.stack([seq_xs, seq_ys], axis=1)
    #     # sort the sequence regarding the ascending order of the temporal sample.
    #     seq_ts = tf.contrib.framework.sort(seq_ts, axis=0, direction='ASCENDING')
    #     return seq_ts, seq_ls, Ns

    def generate(self, T, x_lim, y_lim, batch_size):
        '''
        To generate a homogeneous Poisson point pattern in space S, this function
        uses a two step procedure:
        1. Simulate the number of events n = N(S) occurring in S according to a
        Poisson distribution with mean lam * |S|.
        2. Sample each of the n location according to a uniform distribution on S
        respectively.
        Returns samples: point process samples 
            [(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)]
        '''
        t         = tf.constant(0, dtype=tf.float32)
        loop_vars = [t]
        cond      = lambda t: tf.less(t, T) # termint < T
        body      = lambda t: self.__sampling_point(t)
        return tf.while_loop(cond, body, loop_vars)

    @staticmethod
    def __sampling_point(t):
        return tf.add(t, 1)


        
if __name__ == "__main__":
    kernel = DiffusionKernel()
    hp     = MarkedSpatialTemporalHawkesProcess(kernel)
    
    tf.set_random_seed(1)
    with tf.Session() as sess:
        points = sess.run(hp.generate(T=10.0, x_lim=[0, 1], y_lim=[0, 1], batch_size=2))
        print(points)