#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Imitation Learning for Point Process

A multivariate Hawkes processes model for generating marked spatial-temporal points.

References:
- https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator
- https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf 
- https://www.math.fsu.edu/~ychen/research/multiHawkes.pdf  

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

    def __init__(self, n_nodes, beta=1.):
        """
        Params:
        """
        # model parameters
        self.mu        = tf.get_variable(name="mu", initializer=10 * tf.random_uniform([n_nodes], 0, 1))
        self.alpha     = tf.get_variable(name="alpha", initializer=tf.random_uniform([n_nodes, n_nodes], 0, 1))
        self.beta      = tf.get_variable(name="beta", initializer=tf.random_uniform([n_nodes, n_nodes], 0, 1))
        self.n_nodes   = n_nodes
        # sampling process
        # self._homogeneous_sampling(T=10.0, x_lim=[0, 1], y_lim=[0, 1], batch_size=2)

    def _lam_value(self, t, his_t, k, alpha, beta, mu):
        """
        get lambda intensity value given sample current t, past samples his_t, and 
        model parameters, including alpha, beta, mu. k is the number of components
        that used to caculate the intensity value.
        """
        # k is no larger than n_nodes
        if k >= self.n_nodes:
            k = self.n_nodes
        # intensity value
        val = np.sum([
            mu[i] + np.sum([
                np.sum([
                    alpha[i][j] * np.exp(- beta[i][j] * (t - tau))
                    for tau in his_t ])
                for j in range(self.n_nodes) ])
            for i in range(k) ])
        return val

    def generate_samples(self, sess, T, batch_size=1):
        '''
        generate samples by thinning algorithm, where model parameters are evaluated 
        from tensorflow variables. 
        '''
        # evaluate current model parameters
        mu, alpha, beta = sess.run([self.mu, self.alpha, self.beta])
        seq_ts = []
        seq_ss = []
        for _ in range(batch_size):
            # sequence of time and indices of components
            seq_t = []
            seq_s = []
            # first time and index of component
            t     = 0
            s     = 0
            # generate samples
            # for _ in range(self.n_samples):
            while t < T:
                # upper bound of the intensity lambda
                lam_bar = self._lam_value(t, seq_t, self.n_nodes, alpha, beta, mu)
                u = np.random.uniform()
                w = -np.log(u) / lam_bar
                t = t + w
                if t > T:
                    break
                lam = self._lam_value(t, seq_t, self.n_nodes, alpha, beta, mu)
                D = np.random.uniform()
                # accept the current sample with the probability lam / lam_bar
                if D * lam_bar <= lam:
                    # search for the first k such that D * lam <= lam_k
                    s = self.n_nodes - 1
                    for k in range(self.n_nodes):
                        lam_k = self._lam_value(t, seq_t, k, alpha, beta, mu)
                        if D * lam_bar <= lam_k:
                            s = k + 1
                            break
                    # record the current sample (t, s)
                    seq_t.append(t)
                    seq_s.append(s)
            seq_ts.append(seq_t)
            seq_ss.append(seq_s)
        # organize sequence as tensor
        max_len   = max(map(len, seq_ts))
        mat_seq_t = np.zeros([batch_size, max_len])
        mat_seq_s = np.zeros([batch_size, max_len])
        for b in range(batch_size):
            mat_seq_t[b, :len(seq_ts[b])] = seq_ts[b]
            mat_seq_s[b, :len(seq_ss[b])] = seq_ss[b]
        return mat_seq_t, mat_seq_s

    # def log_likelihood(self, ):


                
        
if __name__ == "__main__":
    hp = MarkedSpatialTemporalMultivariateHawkesProcess(n_nodes=10)
    
    # tf.set_random_seed(0)
    # np.random.seed(0)
    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        seq_t, seq_s = hp.generate_samples(sess, T=10, batch_size=3)
        print(seq_t)
        print(seq_s)