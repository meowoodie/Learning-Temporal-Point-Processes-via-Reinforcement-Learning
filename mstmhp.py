#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Imitation Learning for Point Process

A multivariate Hawkes processes model for generating marked spatial-temporal points.

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

class MarkedSpatialTemporalMultivariateHawkesProcess(object):
    """
    Marked Spatial Temporal Hawkes Process

    A stochastic marked spatial temporal points generator based on Hawkes process.
    """

    def __init__(self, n_samples, n_nodes, beta=1.):
        """
        Params:
        """
        # model parameters
        self.mu        = tf.get_variable(name="mu", initializer=tf.random_normal([n_nodes]))
        self.alpha     = tf.get_variable(name="alpha", initializer=tf.random_normal([n_nodes, n_nodes]))
        self.beta      = tf.get_variable(name="beta", initializer=tf.random_normal([1]))
        self.n_samples = n_samples
        self.n_nodes   = n_nodes
        # generated samples
        self.seq_t   = []
        self.seq_l   = []
        # sampling process
        # self._homogeneous_sampling(T=10.0, x_lim=[0, 1], y_lim=[0, 1], batch_size=2)

    def _lam_value(self, t, his_t, s, his_s):
        '''
        return the intensity value at (t, l).
        The last element of seq_t and seq_s is the location (t, l) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        '''
        if len(his_t) > 0:
            his_t = tf.stack(his_t) # [step_size, 1]
            his_s = tf.stack(his_s) # [step_size, 2]
            val   = self.mu + tf.reduce_sum(self.kernel(t, his_t, s, his_s))
        else:
            val   = self.mu
        return val

    def generate_samples(self, sess, T, batch_size):
        '''
        generate samples by thinning algorithm
        '''
        # evaluate current model parameters
        mu, alpha, beta = sess.run([self.mu, self.alpha, self.beta])
        
        seq_t = []
        seq_s = []
        t     = 0 
        # while s < T:
        for _ in range(self.n_samples):
            # upper bound of the intensity lambda
            lam_bar = np.sum([
                self.mu[i] + np.sum([
                    np.sum(self.alpha[i][j] * np.exp(-self.beta * (s - seq_t))])
                    for j in self.n_nodes ])
                for i in self.n_nodes ])
            u = np.random.uniform()
            w = -np.log(u)/lambda_bar
            s = s + w


            lam_bar = 0
            for i in range(m):
                lambda_bar = lambda_bar + lambda0
                for j in range(m):
                    if A[j,i]>0:
                        for tau in T_m[j]:
                            lambda_bar = lambda_bar + A[j,i] * np.exp(-beta*(s - tau))
            u = np.random.uniform()
            w = -np.log(u)/lambda_bar
            s = s + w
            if s>T:
                break
            lambda_m =[]
            for i in range(m):
                lambda_m.append(lambda0)
                for j in range(m):
                    if A[j,i]>0:
                        for tau in T_m[j]:
                            lambda_m[i] = lambda_m[i] + A[j,i] * np.exp(-beta*(s - tau))
            lambda_m = np.array(lambda_m)
            D = np.random.uniform()
            if D*lambda_bar <= np.sum(lambda_m):
                k = 0
                while D*lambda_bar > np.sum(lambda_m[0:(k+1)]):
                    k = k + 1
                num_events[k] = num_events[k] + 1
                T_m[k].append(s)
                events.append([s,k])
        return(events)


    # def _thinning(self, T, xlim, ylim, batch_size):
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
    #     # def accept_sample(t, l):
    #     #     seq_t.append(t)
    #     #     seq_l.append(l)
    #     #     return t

    #     seq_t  = []
    #     seq_l  = []
    #     seq_c  = []
    #     t      = tf.constant([0.], dtype=tf.float32)
    #     for _ in range(self.n_samples):
    #         lam_hat = self._lam_value(t, seq_t, l, seq_l)
    #         u = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)
    #         d = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)[0] # acceptance rate
    #         w = - tf.log(u) / lam_hat                                           # so that w ~ exponential(lam_hat)
    #         t = t + w                                                           # so that t is the next candidate point
    #         cond = tf.cond(d * lam_hat <= self._lam_value(t, seq_t, l, seq_l), 
    #             lambda: tf.constant(1., dtype=tf.float32),                      
    #             lambda: tf.constant(0., dtype=tf.float32))
    #         seq_c.append(cond)
    #         seq_t.append(t * cond)
    #         seq_l.append(l * cond)

    #     return tf.stack(seq_c)


        
if __name__ == "__main__":
    hp = MarkedSpatialTemporalMultivariateHawkesProcess(step_size=100)
    
    tf.set_random_seed(1)
    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        points = sess.run(hp._homogeneous_sampling(T=1., xlim=[0.,1.], ylim=[0.,1.], batch_size=3))
        print(points)