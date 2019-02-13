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
    def __init__(self, beta=1., C=1., sigma=[3., 3.], offset=[0, 0]):
        self.beta   = tf.get_variable(name="kernel_beta", initializer=tf.constant(beta, dtype=tf.float32))
        self.sigma  = tf.get_variable(name="kernel_sigma", initializer=tf.constant(sigma, dtype=tf.float32))
        self.C      = C

    def nu(self, delta_t, delta_l):
        return (self.C/(2*np.pi*tf.reduce_prod(self.sigma)*delta_t)) * tf.exp(-1*self.beta*delta_t) * \
               tf.exp(-tf.expand_dims(tf.reduce_sum((tf.square(delta_l) / tf.square(self.sigma)), axis=1), -1) / (2*delta_t))

class MarkedSpatialTemporalHawkesProcess(object):
    """
    Marked Spatial Temporal Hawkes Process

    A stochastic marked spatial temporal points generator based on Hawkes process.
    """

    def __init__(self, kernel, step_size, mu=5., alpha=1., beta=1., max_lam=500., m_dim=None):
        """
        Params:
        - T:     maximum time
        - m_dim: number of categories of marks
        """
        # model parameters
        self.mu        = tf.get_variable(name="lam_mu", initializer=tf.constant(mu, dtype=tf.float32))
        self.alpha     = tf.get_variable(name="lam_alpha", initializer=tf.constant(alpha, dtype=tf.float32))
        self.beta      = tf.get_variable(name="lam_beta", initializer=tf.constant(beta, dtype=tf.float32))
        self.max_lam   = max_lam
        self.kernel    = kernel
        self.step_size = step_size
        # generated samples
        self.seq_t   = []
        self.seq_l   = []
        # sampling process
        # self._homogeneous_sampling(T=10.0, x_lim=[0, 1], y_lim=[0, 1], batch_size=2)

    def _lam_value(self, t, his_t, l, his_l):
        '''
        return the intensity value at (t, l).
        The last element of seq_t and seq_s is the location (t, l) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        '''
        if len(his_t) > 0:
            his_t = tf.stack(his_t) # [step_size, 1]
            his_l = tf.stack(his_l) # [step_size, 2]
            val   = self.mu + self.alpha * tf.reduce_sum(self.beta * self.kernel.nu(t-his_t, l-his_l))
        else:
            val   = self.mu
        return val

    def _inhomogeneous_sampling(self, T, xlim, ylim, batch_size):
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
        # def accept_sample(t, l):
        #     seq_t.append(t)
        #     seq_l.append(l)
        #     return t

        seq_t  = []
        seq_l  = []
        seq_c  = []
        t_plus = tf.constant([10.], dtype=tf.float32)
        t      = tf.constant([0.], dtype=tf.float32)
        l      = tf.random_uniform([2], minval=xlim[0], maxval=xlim[1], dtype=tf.float32)
        for _ in range(self.step_size):
            lam_hat = self._lam_value(t_plus, seq_t, l, seq_l)
            u = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)                  
            l = tf.random_uniform([2], minval=xlim[0], maxval=xlim[1])          # generated location sample
            d = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)[0] # acceptance rate
            w = - tf.log(u) / lam_hat                                           # so that w ~ exponential(lam_hat)
            t = t + w                                                           # so that t is the next candidate point
            cond = tf.cond(d * lam_hat <= self._lam_value(t, seq_t, l, seq_l), 
                lambda: tf.constant(1., dtype=tf.float32),                      
                lambda: tf.constant(0., dtype=tf.float32))
            seq_c.append(cond)
            seq_t.append(t * cond)
            seq_l.append(l * cond)

        return tf.stack(seq_c)

        
if __name__ == "__main__":
    kernel = DiffusionKernel()
    hp     = MarkedSpatialTemporalHawkesProcess(kernel=kernel, step_size=100)
    
    tf.set_random_seed(1)
    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        points = sess.run(hp._homogeneous_sampling(T=1., xlim=[0.,1.], ylim=[0.,1.], batch_size=3))
        print(points)

        # print(sess.run([ hp.alpha, hp.beta, hp.mu ]))
