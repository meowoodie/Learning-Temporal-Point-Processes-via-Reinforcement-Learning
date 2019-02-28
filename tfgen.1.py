#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Imitation Learning for Point Process

A LSTM based model for generating marked spatial-temporal points.

References:
- https://arxiv.org/abs/1811.05016

Dependencies:
- Python 3.6.7
- tensorflow==1.5.0
"""

import sys
import arrow
import utils
import numpy as np
import tensorflow as tf

from stppg import DiffusionKernel, HawkesLam, SpatialTemporalPointProcess

class SpatialTemporalHawkes(object):
    """
    """

    def __init__(self, C=1., maximum=1e+4, verbose=False):
        """
        """
        INIT_PARAM   = 5e-2
        self.C       = C       # constant in kernel function
        self.maximum = maximum # upper bound of conditional intensity
        self.mu      = tf.get_variable(name="mu", initializer=tf.constant(0.1), dtype=tf.float32)
        self.beta    = tf.get_variable(name="beta", initializer=tf.constant(1.), dtype=tf.float32)
        self.sigma_x = tf.get_variable(name="sigma_x", initializer=tf.constant(0.1), dtype=tf.float32)
        self.sigma_y = tf.get_variable(name="sigma_y", initializer=tf.constant(0.1), dtype=tf.float32)
        self.verbose = verbose

    def _kernel(self, x, y, t):
        """
        difussion kernel function proposed by Musmeci and Vere-Jones (1992).
        """
        return (self.C / (2 * np.pi * self.sigma_x * self.sigma_y * t)) * \
               tf.exp(- self.beta * t - (tf.square(x)/tf.square(self.sigma_x) + tf.square(y)/tf.square(self.sigma_y)) / (2*t))

    def _lambda(self, x, y, t, x_his, y_his, t_his):
        """
        lambda function for the Hawkes process.
        """
        lam = self.mu + tf.reduce_sum(self._kernel(x - x_his, y - y_his, t - t_his), axis=0)
        return lam

    @staticmethod
    def __homogeneous_poisson_sampling(T, S, maximum):
        """
        To generate a homogeneous Poisson point pattern in space S X T, it basically
        takes two steps:
        1. Simulate the number of events n = N(S) occurring in S according to a
        Poisson distribution with mean lam * |S X T|.
        2. Sample each of the n location according to a uniform distribution on S
        respectively.

        Args:
            lam: intensity (or maximum intensity when used by thining algorithm)
            S:   [(min_t, max_t), (min_x, max_x), (min_y, max_y), ...] indicates the
                range of coordinates regarding a square (or cubic ...) region.
        Returns:
            samples: point process samples:
            [(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)]
        """
        _S = [T] + S
        # sample the number of events from S
        n  = utils.lebesgue_measure(_S)
        N  = tf.random.poisson(lam=maximum * n, shape=[1], dtype=tf.int32)
        # simulate spatial sequence and temporal sequence separately.
        points    = [ tf.random.uniform(shape=N, minval=_S[i][0], maxval=_S[i][1]) for i in range(len(_S)) ]
        # sort the temporal sequence ascendingly.
        points[0] = tf.contrib.framework.sort(points[0], direction="ASCENDING")
        points    = tf.transpose(tf.stack(points))
        return points

    def _inhomogeneous_poisson_thinning(self, homo_points, maximum):
        """
        To generate a realization of an inhomogeneous Poisson process in S × T, this
        function uses a thining algorithm as follows. For a given intensity function
        lam(s, t):
        1. Define an upper bound max_lam for the intensity function lam(s, t)
        2. Simulate a homogeneous Poisson process with intensity max_lam.
        3. "Thin" the simulated process as follows,
            a. Compute p = lam(s, t)/max_lam for each point (s, t) of the homogeneous
            Poisson process
            b. Generate a sample u from the uniform distribution on (0, 1)
            c. Retain the locations for which u <= p.
        """
        # number of home points
        n_homo_points = tf.shape(homo_points)[0]

        # thining procedure
        # - input:  current index of homo points & current selection for retained points
        # - return: updated selection for retained points
        def thining(i, selection):
            retained_points     = tf.boolean_mask(homo_points, selection)
            x, y, t             = homo_points[i, 1], homo_points[i, 2], homo_points[i, 0]
            his_x, his_y, his_t = retained_points[:, 1], retained_points[:, 2], retained_points[:, 0]
            # thinning
            lam_value = self._lambda(x, y, t, his_x, his_y, his_t)
            lam_bar   = maximum
            D         = tf.random.uniform(shape=[1], minval=0., maxval=1.)[0]
            # accept: return this point
            upd_selection = selection
            upd_selection = tf.add(upd_selection, tf.one_hot(i, n_homo_points))
            # reject: return zero entry
            return tf.cond(tf.less(D * lam_bar, lam_value), 
                lambda: upd_selection, # retain this point
                lambda: selection)     # return the same selection without any change

        # get thining selection
        selections = tf.scan(
            lambda selection, i: thining(i, selection),
            tf.range(n_homo_points),                                         # indices of homo points
            initializer=(tf.zeros(shape=[n_homo_points], dtype=tf.float32))) # initial selection
        # get retained points
        retained_points = tf.boolean_mask(homo_points, selections[-1, :])
        return retained_points
    
    def sampling(self, T, S, maximum, batch_size, keep_latest_k):
        """
        generate samples with batch_size by thining algorithm, return sampling sequences and 
        corresponding element-wise loglikelihood value.
        """
        points_list = []
        size_list   = []
        # generate inhomogeneous poisson points iterately
        for b in range(batch_size):
            homo_points = self.__homogeneous_poisson_sampling(T, S, maximum)
            points      = self._inhomogeneous_poisson_thinning(homo_points, maximum)
            n_points    = tf.shape(points)[0]
            points_list.append(points)
            size_list.append(n_points)
        # initialize tensor for sequences
        max_size = tf.reduce_max(tf.stack(size_list))
        seqs     = []
        logliks  = []
        # organize generated samples into tensor seqs
        for b in range(batch_size):
            n_points     = tf.shape(points_list[b])[0]
            points       = points_list[b]
            logpdfs      = tf.scan(
                lambda a, i: self.log_conditional_pdf(points[:i, :], S, keep_latest_k),
                tf.range(1, n_points+1), # from the first point to the last point
                initializer=np.array(0., dtype=np.float32))
            seq_paddings = tf.zeros((max_size - n_points, 1 + len(S)))
            lik_paddings = tf.zeros(max_size - n_points)
            seq          = tf.concat([points, seq_paddings], axis=0)
            loglik       = tf.concat([logpdfs, lik_paddings], axis=0)
            seqs.append(seq)
            logliks.append(loglik)
        seqs    = tf.stack(seqs, axis=0)
        logliks = tf.stack(logliks, axis=0)
        return seqs, logliks

    def log_conditional_pdf(self, points, S, keep_latest_k=None):
        """
        log pdf conditional of a data point given its history, where the data point is 
        points[-1], and its history is points[:-1]
        """
        if keep_latest_k is not None: 
            points          = points[-keep_latest_k:, :]
        # number of the points
        len_points          = tf.shape(points)[0]
        # variables for calculating triggering probability
        x, y, t             = points[-1, 1],  points[-1, 2],  points[-1, 0]
        x_his, y_his, t_his = points[:-1, 1], points[:-1, 2], points[:-1, 0]

        def pdf_no_history():
            return tf.log(self._lambda(x, y, t, x_his, y_his, t_his))
        
        def pdf_with_history():
            # triggering probability
            log_trig_prob = tf.log(self._lambda(x, y, t, x_his, y_his, t_his))
            # variables for calculating tail probability
            tn, ti        = points[-2, 0], points[:-1, 0]
            t_ti, tn_ti   = t - ti, tn - ti
            # tail probability
            log_tail_prob = - \
                self.mu * (t - t_his[-1]) * utils.lebesgue_measure(S) - \
                tf.reduce_sum(tf.scan(
                    lambda a, i: self.C * (tf.exp(- self.beta * tn_ti[i]) - tf.exp(- self.beta * t_ti[i])) / self.beta,
                    tf.range(tf.shape(t_ti)[0]),
                    initializer=np.array(0., dtype=np.float32)))
            return log_trig_prob + log_tail_prob

        # TODO: Unsolved issue:
        #       pdf_with_history will still be called even if the condition is true, which leads to exception
        #       "ValueError: slice index -1 of dimension 0 out of bounds." due to that points is empty but we 
        #       try to index a nonexisted element.
        #       However, when points is indexed in a scan loop, this works fine and the numerical result is 
        #       also correct. which is very confused to me. Therefore, I leave this problem here temporarily.
        log_cond_pdf = tf.cond(tf.less(len_points, 2), 
            pdf_no_history,   # if there is only one point in the sequence
            pdf_with_history) # if there is more than one point in the sequence
        return log_cond_pdf

if __name__ == "__main__":
    # Unittest example
    tf.random.set_random_seed(1234)
    with tf.Session() as sess:
        hawkes = SpatialTemporalHawkes()

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # homo_points = hawkes._homogeneous_poisson_sampling(T=[0., 10.], S=[[-1., 1.], [-1., 1.]], maximum=1e+3)
        # points      = hawkes._inhomogeneous_poisson_thinning(homo_points, maximum=1e+3)
        seqs, logliks = hawkes.sampling(T=[0., 10.], S=[[-1., 1.], [-1., 1.]], maximum=1e+3, batch_size=3, keep_latest_k=None)
        res1, res2 = sess.run([ seqs, logliks ])
        print(res1, res2)
        print(res1.shape, res2.shape)
