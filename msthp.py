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

    def __init__(self, kernel, step_size, mu=0, alpha=1., beta=1., max_lam=500., m_dim=None):
        """
        Params:
        - T:     maximum time
        - m_dim: number of categories of marks
        """
        # model parameters
        self.mu        = tf.get_variable(name="lam_mu", initializer=tf.constant(mu))
        self.alpha     = tf.get_variable(name="lam_alpha", initializer=tf.constant(alpha))
        self.beta      = tf.get_variable(name="lam_beta", initializer=tf.constant(beta))
        self.max_lam   = max_lam
        self.kernel    = kernel
        self.step_size = step_size
        # generated samples
        self.seq_t   = []
        self.seq_l   = []
        # sampling process
        self._homogeneous_sampling(T=10.0, x_lim=[0, 1], y_lim=[0, 1], batch_size=2)

    def _lam_value(self, seq_t, seq_l):
        '''
        return the intensity value at (t, l).
        The last element of seq_t and seq_s is the location (t, l) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        '''
        seq_t = tf.stack(self.seq_t, axis=1) # [batch_size, step_size, 1]
        seq_l = tf.stack(self.seq_l, axis=1) # [batch_size, step_size, 2]
        if tf.shape(seq_t)[0] > 1:
            # get current time, spatial values and historical time, spatial values.
            cur_t, his_t = seq_t[-1], seq_t[:-1]
            cur_l, his_l = seq_l[-1], seq_l[:-1]
            val = self.mu + self.alpha * tf.reduce_sum(self.beta * self.kernel.nu(cur_t-his_t, cur_l-his_l))
        else:
            val = self.mu
        return val

    def _homogeneous_sampling(self, T, x_lim, y_lim, batch_size):
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
        # t         = tf.constant(0, dtype=tf.float32)  # initial time
        # loop_vars = [t]                               # loop variables
        # cond      = lambda t: tf.less(t, T)           # termination condition of the loop
        # body      = lambda t: self._sampling_point(t)
        # tf.while_loop(cond, body, loop_vars)
        for _ in range(len(self.step_size)):
            lam_hat = self._lam_value(self.seq_t, self.seq_l)

    def _sampling_point(self, t):
        # lam_hat = self._lam_value(self.seq_t, self.seq_l)
        # # generate sample candidate
        # u = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)
        # w = -tf.log(u) / lam_hat  # so that w ~ exponential(lam_hat)
        # t = t + w                 # so that t is the next candidate point
        # # thinning sample
        # d = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32) # acceptance rate
        # tf.cond(tf.less(d * lam_hat, self._lam_value(self.seq_t, self.seq_l)), )

        
if __name__ == "__main__":
    kernel = DiffusionKernel()
    hp     = MarkedSpatialTemporalHawkesProcess(kernel)
    
    tf.set_random_seed(1)
    with tf.Session() as sess:
        points = sess.run(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32))
        print(points)