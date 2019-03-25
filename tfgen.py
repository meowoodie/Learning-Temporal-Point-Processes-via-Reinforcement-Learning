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

from stppg import GaussianMixtureDiffusionKernel, HawkesLam, SpatialTemporalPointProcess

class SpatialTemporalHawkes(object):
    """
    Customized Spatial Temporal Hawkes

    A Hawkes model parametrized by multi-layers neural networks, which provides flexible self-exciting 
    points pattern.
    """

    def __init__(self, T, S, layers=[20, 20], n_comp=5, C=1., maximum=1e+3, verbose=False):
        """
        """
        # constant hyper parameters
        self.INIT_PARAM  = .01
        self.SIGMA_SHIFT = .05
        self.SIGMA_SCALE = .2
        self.MU_SCALE    = .1
        # configurations
        self.C       = C       # constant
        self.T       = T       # time space
        self.S       = S       # location space
        self.maximum = maximum # upper bound of conditional intensity
        self.verbose = verbose
        # model parameters
        self.mu      = tf.get_variable(name="mu", initializer=tf.constant(0.1), dtype=tf.float32)
        self.beta    = tf.get_variable(name="beta", initializer=tf.constant(1.), dtype=tf.float32)
        self.Wss     = []
        self.bss     = []
        self.Wphis   = []
        # construct multi-layers neural networks
        # - define the layers where 2 is for the input layer (x and y); 
        #   And 5 is for the output layer (mu_x, mu_y, sigma_x, sigma_y, rho)
        self.layers = [2] + layers + [5]
        # - define the number of the components in Gaussian mixture diffusion kernel
        self.n_comp = n_comp
        # - construct component weighting vectors
        for k in range(self.n_comp):
            Wphi = tf.get_variable(name="Wphi%d" % k, 
                initializer=self.INIT_PARAM * tf.random.normal(shape=[2, 1]),
                dtype=tf.float32)
            self.Wphis.append(Wphi)
            # - construct weight & bias matrix layer by layer for each of Gaussian components
            Ws = []
            bs = []
            for i in range(len(self.layers)-1):
                # random initialization
                W = tf.get_variable(name="W%d%d" % (k, i), 
                    initializer=self.INIT_PARAM * tf.random.normal(shape=[self.layers[i], self.layers[i+1]]),
                    dtype=tf.float32)
                b = tf.get_variable(name="b%d%d" % (k, i), 
                    initializer=self.INIT_PARAM * tf.random.normal(shape=[self.layers[i+1]]),
                    dtype=tf.float32)
                Ws.append(W)
                bs.append(b)
            self.Wss.append(Ws)
            self.bss.append(bs)

    def sampling(self, sess, batch_size):
        """fetch model parameters, and generate samples accordingly."""
        # get current model parameters
        mu, beta = sess.run([self.mu, self.beta])
        Wss      = sess.run(self.Wss)
        bss      = sess.run(self.bss)
        Wphis    = sess.run(self.Wphis)
        # construct kernel function and conditional intensity lambda
        kernel   = GaussianMixtureDiffusionKernel(
            self.n_comp, layers=self.layers[1:-1], beta=beta, C=self.C, 
            SIGMA_SHIFT=self.SIGMA_SHIFT, SIGMA_SCALE=self.SIGMA_SCALE, MU_SCALE=self.MU_SCALE,
            Wss=Wss, bss=bss, Wphis=Wphis)
        lam      = HawkesLam(mu, kernel, maximum=self.maximum)
        # sampling points given model parameters
        pp       = SpatialTemporalPointProcess(lam)
        seqs, sizes = pp.generate(T=self.T, S=self.S, batch_size=batch_size, verbose=self.verbose)
        return seqs

    def _nonlinear_mapping(self, k, s):
        """nonlinear mapping from location space to parameters space"""
        # construct multi-layers neural networks
        output = s # [n_his, 2]
        for i in range(len(self.layers)-1):
            output = tf.nn.sigmoid(tf.nn.xw_plus_b(output, self.Wss[k][i], self.bss[k][i])) # [n_his, n_b]
        # project to parameters space
        mu_x    = (output[:, 0] - 0.5) * 2 * self.MU_SCALE           # [n_his]: mu_x spans (-MU_SCALE, MU_SCALE)
        mu_y    = (output[:, 1] - 0.5) * 2 * self.MU_SCALE           # [n_his]: mu_y spans (-MU_SCALE, MU_SCALE)
        sigma_x = output[:, 2] * self.SIGMA_SCALE + self.SIGMA_SHIFT # [n_his]: sigma_x spans (SIGMA_SHIFT, SIGMA_SHIFT + SIGMA_SCALE)
        sigma_y = output[:, 3] * self.SIGMA_SCALE + self.SIGMA_SHIFT # [n_his]: sigma_y spans (SIGMA_SHIFT, SIGMA_SHIFT + SIGMA_SCALE)
        rho     = output[:, 4] * 1.5 - .75                           # [n_his]: rho spans (-.75, .75)
        return mu_x, mu_y, sigma_x, sigma_y, rho

    def _gaussian_kernel(self, k, t, s, his_t, his_s):
        """
        A Gaussian diffusion kernel function based on the standard kernel function proposed 
        by Musmeci and Vere-Jones (1992). The angle and shape of diffusion ellipse is able  
        to vary according to the location. 

        k indicates the k-th gaussian component that is used to compute the nonlinear mappings.  
        """
        eps     = 1e-8            # IMPORTANT: Avoid delta_t be zero
        delta_t = t - his_t + eps # [n_his]
        delta_s = s - his_s       # [n_his, 2]
        delta_x = delta_s[:, 0]   # [n_his]
        delta_y = delta_s[:, 1]   # [n_his]
        mu_x, mu_y, sigma_x, sigma_y, rho = self._nonlinear_mapping(k, his_s)
        return tf.exp(- self.beta * delta_t) * \
            (self.C / (2 * np.pi * sigma_x * sigma_y * delta_t * tf.sqrt(1 - tf.square(rho)))) * \
            tf.exp((- 1. / (2 * delta_t * (1 - tf.square(rho)))) * \
                ((tf.square(delta_x - mu_x) / tf.square(sigma_x)) + \
                (tf.square(delta_y - mu_y) / tf.square(sigma_y)) - \
                (2 * rho * (delta_x - mu_x) * (delta_y - mu_y) / (sigma_x * sigma_y))))
    
    def _softmax(self, s, k):
        """
        Gaussian mixture components are weighted by phi^k, which are computed by a softmax function, i.e., 
        phi^k(x, y) = e^{[x y]^T w^k} / \sum_{i=1}^K e^{[x y]^T w^i}
        """
        # s:        [n_his, 2]
        # Wphis[k]: [2, 1]
        numerator   = tf.exp(tf.matmul(s, self.Wphis[k]))                        # [n_his, 1]
        denominator = tf.concat([ 
            tf.exp(tf.matmul(s, self.Wphis[i])) 
            for i in range(self.n_comp) ], axis=1)                               # [n_his, K=n_comp]
        phis        = tf.squeeze(numerator) / tf.reduce_sum(denominator, axis=1) # [n_his]
        return phis
    
    def _gaussian_mixture_kernel(self, t, s, his_t, his_s):
        """
        A Gaussian mixture diffusion kernel function is superposed by multiple Gaussian diffusion 
        kernel function. The number of the Gaussian components is specified by n_comp. 
        """
        nus = []
        for k in range(self.n_comp):
            phi = self._softmax(his_s, k)                            # [n_his]             
            nu  = phi * self._gaussian_kernel(k, t, s, his_t, his_s) # [n_his]
            nu  = tf.expand_dims(nu, -1)                             # [n_his, 1]
            nus.append(nu)                                           # K * [n_his, 1]
        nus = tf.concat(nus, axis=1)      # [n_his, K]
        return tf.reduce_sum(nus, axis=1) # [n_his]

    def _lambda(self, t, s, his_t, his_s):
        """lambda function for the Hawkes process."""
        lam = self.mu + tf.reduce_sum(self._gaussian_kernel(0, t, s, his_t, his_s), axis=0)
        return lam

    def log_conditional_pdf(self, points, keep_latest_k=None):
        """log pdf conditional on history."""
        if keep_latest_k is not None: 
            points          = points[-keep_latest_k:, :]
        # number of the points
        len_points          = tf.shape(points)[0]
        # variables for calculating triggering probability
        # x, y, t             = points[-1, 1],  points[-1, 2],  points[-1, 0]
        # x_his, y_his, t_his = points[:-1, 1], points[:-1, 2], points[:-1, 0]
        s, t         = points[-1, 1:], points[-1, 0]
        his_s, his_t = points[:-1, 1:], points[:-1, 0]

        def pdf_no_history():
            return tf.log(tf.clip_by_value(self._lambda(t, s, his_t, his_s), 1e-8, 1e+10))
        
        def pdf_with_history():
            # triggering probability
            log_trig_prob = tf.log(tf.clip_by_value(self._lambda(t, s, his_t, his_s), 1e-8, 1e+10))
            # variables for calculating tail probability
            tn, ti        = points[-2, 0], points[:-1, 0]
            t_ti, tn_ti   = t - ti, tn - ti
            # tail probability
            log_tail_prob = - \
                self.mu * (t - his_t[-1]) * utils.lebesgue_measure(self.S) - \
                tf.reduce_sum(tf.scan(
                    lambda a, i: self.C * (tf.exp(- self.beta * tn_ti[i]) - tf.exp(- self.beta * t_ti[i])) / \
                        tf.clip_by_value(self.beta, 1e-8, 1e+10),
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

    def save_params_npy(self, sess, path):
        """save parameters into numpy file."""
        Wss      = sess.run(self.Wss)
        bss      = sess.run(self.bss)
        Wphis    = sess.run(self.Wphis)
        mu, beta = sess.run([self.mu, self.beta])
        np.savez(path, Wss=Wss, bss=bss, Wphis=Wphis, mu=mu, beta=beta)



class MarkedSpatialTemporalLSTM(object):
    """
    Customized Stochastic LSTM Network

    A LSTM Network with customized stochastic output neurons, which used to generate time, location and marks accordingly.
    """

    def __init__(self, step_size, lstm_hidden_size, loc_hidden_size, mak_hidden_size, m_dim, x_lim=5, y_lim=5, epsilon=0.3):
        """
        Params:
        - step_size:        the steps (length) of the LSTM network
        - lstm_hidden_size: size of hidden state of the LSTM
        - loc_hidden_size:  size of hidden feature of location
        - mak_hidden_size:  size of hidden feature of mark
        - m_dim:            number of categories of marks
        """
        
        # data dimension
        self.t_dim = 1     # by default
        self.m_dim = m_dim # number of categories for the marks

        # model hyper-parameters
        self.step_size         = step_size        # step size of LSTM
        self.lstm_hidden_size  = lstm_hidden_size # size of LSTM hidden feature
        self.loc_hidden_size   = loc_hidden_size  # size of location hidden feature
        self.loc_param_size    = 5                # by default
        self.mak_hidden_size   = mak_hidden_size  # size of mark hidden feature
        self.x_lim, self.y_lim = x_lim, y_lim
        self.epsilon           = epsilon

        INIT_PARAM_RATIO = 1 / np.sqrt(self.loc_hidden_size * self.loc_param_size)

        # define learning weights
        # - time weights
        self.Wt  = tf.get_variable(name="Wt", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.t_dim]))
        self.bt  = tf.get_variable(name="bt", initializer=INIT_PARAM_RATIO * tf.random_normal([self.t_dim]))
        # - location weights
        self.Wl0 = tf.get_variable(name="Wl0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.loc_hidden_size]))
        self.bl0 = tf.get_variable(name="bl0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.loc_hidden_size]))
        self.Wl1 = tf.get_variable(name="Wl1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.loc_hidden_size, self.loc_param_size]))
        self.bl1 = tf.get_variable(name="bl1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.loc_param_size]))
        # - mark weights
        self.Wm0 = tf.get_variable(name="Wm0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.lstm_hidden_size, self.mak_hidden_size]))
        self.bm0 = tf.get_variable(name="bm0", initializer=INIT_PARAM_RATIO * tf.random_normal([self.mak_hidden_size]))
        self.Wm1 = tf.get_variable(name="Wm1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.mak_hidden_size, self.m_dim]))
        self.bm1 = tf.get_variable(name="bm1", initializer=INIT_PARAM_RATIO * tf.random_normal([self.m_dim]))

    def initialize_network(self, batch_size):
        """Create a new network for training purpose, where the LSTM is at the zero state"""
        # create a basic LSTM cell
        tf_lstm_cell    = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size)
        # defining initial basic LSTM hidden state [2, batch_size, lstm_hidden_size]
        # - lstm_state.h: hidden state [batch_size, lstm_hidden_size]
        # - lstm_state.c: cell state   [batch_size, lstm_hidden_size]
        init_lstm_state = tf_lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # construct customized LSTM network
        self.seq_t, self.seq_l, self.seq_m, self.seq_loglik, self.final_state = self._recurrent_structure(
            batch_size, tf_lstm_cell, init_lstm_state)

    def _recurrent_structure(self, 
            batch_size, 
            tf_lstm_cell,     # tensorflow LSTM cell object, e.g. 'tf.nn.rnn_cell.BasicLSTMCell'
            init_lstm_state): # initial LSTM state tensor
        """Recurrent structure with customized LSTM cells."""
        # defining initial data point
        # - init_t: initial time     [batch_size, t_dim] 
        init_t = tf.zeros([batch_size, self.t_dim], dtype=tf.float32)
        # concatenate each customized LSTM cell by loop
        seq_t      = [] # generated sequence initialization
        seq_l      = []
        seq_m      = []
        seq_loglik = []
        last_t, last_lstm_state = init_t, init_lstm_state # loop initialization
        for _ in range(self.step_size):
            t, l, m, loglik, state = self._customized_lstm_cell(batch_size, tf_lstm_cell, last_lstm_state, last_t)
            seq_t.append(t)           # record generated time
            seq_l.append(l)           # record generated location
            seq_m.append(m)           # record generated mark 
            seq_loglik.append(loglik) # record log likelihood  
            last_t          = t       # reset last_t
            last_lstm_state = state   # reset last_lstm_state
        seq_t      = tf.stack(seq_t, axis=1)      # [batch_size, step_size, t_dim]
        seq_l      = tf.stack(seq_l, axis=1)      # [batch_size, step_size, 2]
        seq_m      = tf.stack(seq_m, axis=1)      # [batch_size, step_size, m_dim]
        seq_loglik = tf.stack(seq_loglik, axis=1) # [batch_size, step_size, 1]
        return seq_t, seq_l, seq_m, seq_loglik, state

    def _customized_lstm_cell(self, batch_size, 
            tf_lstm_cell, # tensorflow LSTM cell object, e.g. 'tf.nn.rnn_cell.BasicLSTMCell'
            last_state,   # last state as input of this LSTM cell
            last_t):      # last_t + delta_t as input of this LSTM cell
        """
        Customized Stochastic LSTM Cell

        The customized LSTM cell takes current (time 't', location 'l', mark 'm') and the hidden state of last moment
        as input, return the ('next_t', 'next_l', 'next_m') as well as the hidden state for the next moment. The time,
        location and mark will be sampled based upon last hidden state.

        The reason avoid using tensorflow builtin rnn structure is that, besides last hidden state, the other feedback 
        to next moment is a customized stochastic variable which depends on the last moment's rnn output. 
        """
        # stochastic neurons for generating time, location and mark
        delta_t, loglik_t = self._dt(batch_size, last_state.h) # [batch_size, t_dim], [batch_size, 1] 
        next_l,  loglik_l = self._l(batch_size, last_state.h)  # [batch_size, 2],     [batch_size, 1] 
        next_m,  loglik_m = self._m(batch_size, last_state.h)  # [batch_size, m_dim], [batch_size, 1]  
        next_t = last_t + delta_t                              # [batch_size, t_dim]
        # log likelihood
        loglik = loglik_l # + loglik_l # + loglik_m    # TODO: Add mark to input x
        # input of LSTM
        x      = tf.concat([next_l], axis=1) # TODO: Add mark to input x
        # one step rnn structure
        # - x is a tensor that contains a single step of data points with shape [batch_size, t_dim + l_dim + m_dim]
        # - state is a tensor of hidden state with shape [2, batch_size, state_size]
        _, next_state = tf.nn.static_rnn(tf_lstm_cell, [x], initial_state=last_state, dtype=tf.float32)
        return next_t, next_l, next_m, loglik, next_state

    def _dt(self, batch_size, hidden_state):
        """Sampling time interval given hidden state of LSTM"""
        theta_h = tf.nn.elu(tf.matmul(hidden_state, self.Wt) + self.bt) + 1                         # [batch_size, t_dim=1]
        # reparameterization trick for sampling action from exponential distribution
        delta_t = - tf.log(tf.random_uniform([batch_size, self.t_dim], dtype=tf.float32)) / theta_h # [batch_size, t_dim=1]
        # log likelihood
        loglik  = - tf.multiply(theta_h, delta_t) + tf.log(theta_h)                                 # [batch_size, 1]
        return delta_t, loglik

    def _l(self, batch_size, hidden_state):
        """Sampling location shifts given hidden state of LSTM"""
        # masks for epsilon greedy exploration & regular sampling
        p = tf.random_uniform([batch_size, 1], 0, 1)                  # [batch_size, 1]
        l_eps_mask = tf.cast(p < self.epsilon, dtype=tf.float32)      # [batch_size, 1]
        l_reg_mask = 1. - l_eps_mask                                  # [batch_size, 1]

        # sample from uniform distribution (epsilon greedy exploration)
        lx_eps = tf.random_uniform([batch_size, 1], minval=-self.x_lim, maxval=self.x_lim, dtype=tf.float32)
        ly_eps = tf.random_uniform([batch_size, 1], minval=-self.y_lim, maxval=self.y_lim, dtype=tf.float32)

        # sample from the distribution detemined by hidden state
        dense_feature = tf.nn.relu(tf.matmul(hidden_state, self.Wl0)) + self.bl0  # [batch_size, loc_hidden_size]
        dense_feature = tf.matmul(dense_feature, self.Wl1) + self.bl1             # [batch_size, loc_param_size]
        # - 5 params that determine the distribution of location shifts with shape [batch_size]
        mu0 = tf.reshape(dense_feature[:, 0], [batch_size, 1]) 
        mu1 = tf.reshape(dense_feature[:, 1], [batch_size, 1])
        # - construct positive definite and symmetrical matrix as covariance matrix
        A11 = tf.expand_dims(tf.reshape(dense_feature[:, 2], [batch_size, 1]), -1) # [batch_size, 1, 1]
        A22 = tf.expand_dims(tf.reshape(dense_feature[:, 3], [batch_size, 1]), -1) # [batch_size, 1, 1]
        A21 = tf.expand_dims(tf.reshape(dense_feature[:, 4], [batch_size, 1]), -1) # [batch_size, 1, 1]
        A12 = tf.zeros([batch_size, 1, 1])                                         # [batch_size, 1, 1]
        A1  = tf.concat([A11, A12], axis=2) # [batch_size, 1, 2]
        A2  = tf.concat([A21, A22], axis=2) # [batch_size, 1, 2]
        A   = tf.concat([A1, A2], axis=1)   # [batch_size, 2, 2]
        # - sigma = A * A^T with shape [batch_size, 2, 2]
        sigma   = tf.scan(lambda a, x: tf.matmul(x, tf.transpose(x)), A) # [batch_size, 2, 2]
        sigma11 = tf.expand_dims(sigma[:, 0, 0], -1)                     # [batch_size, 1]
        sigma22 = tf.expand_dims(sigma[:, 1, 1], -1)                     # [batch_size, 1]
        sigma12 = tf.expand_dims(sigma[:, 0, 1], -1)                     # [batch_size, 1]
        # - random variable for generating locaiton
        rv0 = tf.random_normal([batch_size, 1])
        rv1 = tf.random_normal([batch_size, 1])
        # - location x and y
        x = mu0 + tf.multiply(sigma11, rv0) + tf.multiply(sigma12, rv1) # [batch_size, 1]
        y = mu1 + tf.multiply(sigma12, rv0) + tf.multiply(sigma22, rv1) # [batch_size, 1]

        # # combine exploration and regular sampling
        # x = tf.multiply(lx_eps, l_eps_mask) + tf.multiply(x, l_reg_mask)
        # y = tf.multiply(ly_eps, l_eps_mask) + tf.multiply(y, l_reg_mask)
        l = tf.concat([x, y], axis=1)                         # [batch_size, 2]

        # log likelihood
        sigma1 = tf.sqrt(tf.square(sigma11) + tf.square(sigma12))
        sigma2 = tf.sqrt(tf.square(sigma12) + tf.square(sigma22))
        v12 = tf.multiply(sigma11, sigma12) + tf.multiply(sigma12, sigma22)
        rho = v12 / tf.multiply(sigma1, sigma2)
        z   = tf.square(x - mu0) / tf.square(sigma1) \
            - 2 * tf.multiply(rho, tf.multiply(x - mu0, y - mu1)) / tf.multiply(sigma1, sigma2) \
            + tf.square(y - mu1) / tf.square(sigma2)
        loglik = - z / 2 / (1 - tf.square(rho)) \
                 - tf.log(2 * np.pi * tf.multiply(tf.multiply(sigma1, sigma2), tf.sqrt(1 - tf.square(rho))))
                 
        return l, loglik
    
    def _m(self, batch_size, hidden_state):
        """Sampling mark given hidden state of LSTM"""
        dense_feature = tf.nn.relu(tf.matmul(hidden_state, self.Wm0)) + self.bm0      # [batch_size, location_para_dim]
        dense_feature = tf.nn.elu(tf.matmul(dense_feature, self.Wm1) + self.bm1) + 1  # [batch_size, dim_m] dense_feature is positive
        # sample from multinomial distribution (use Gumbel trick to sample the labels)
        eps        = 1e-13
        rv_uniform = tf.random_uniform([batch_size, self.m_dim])
        rv_Gumbel  = -tf.log(-tf.log(rv_uniform + eps) + eps)
        label      = tf.argmax(dense_feature + rv_Gumbel, axis=1) # label: [batch]
        m          = tf.one_hot(indices=label, depth=self.m_dim)  # [batch_size, m_dim]
        # log likelihood
        prob       = tf.nn.softmax(dense_feature)
        loglik     = tf.log(tf.reduce_sum(m * prob, 1) + 1e-13)
        return m, loglik

if __name__ == "__main__":
    # Unittest example
    np.random.seed(1)
    tf.set_random_seed(1)

    with tf.Session() as sess:
        hawkes = SpatialTemporalHawkes(
            T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
            layers=[5], n_comp=5, C=1., maximum=1e+3, verbose=True)

        points = tf.constant([
            [ 1.16898147e-02,  1.45831794e-01, -3.05314839e-01],
            [ 4.81481478e-02, -1.25229925e-01,  8.72766301e-02],
            [ 1.13194443e-01, -3.87020826e-01,  2.80696362e-01],
            [ 1.60300925e-01, -2.42807735e-02, -5.64230382e-01],
            [ 1.64004624e-01,  7.10764453e-02, -1.77927762e-01],
            [ 1.64236113e-01,  6.51166216e-02, -6.82414293e-01],
            [ 2.05671296e-01, -4.48017061e-01,  5.36620915e-01],
            [ 2.12152779e-01, -3.20064761e-02, -2.08911732e-01]], dtype=tf.float32) 

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # t = points[-1, 0]
        # s = points[-1, 1:]
        # his_t = points[:-1, 0]
        # his_s = points[:-1, 1:]

        # res = sess.run(hawkes.log_conditional_pdf(points))
        # res = sess.run(hawkes._lambda(t, s, his_t, his_s))
        # res = sess.run(hawkes._softmax(his_s, 0))
        # res = sess.run(hawkes._gaussian_kernel(0, t, s, his_t, his_s))

        # test log conditional pdf
        r = tf.scan(
            lambda a, i: hawkes.log_conditional_pdf(points[:i, :]),
            tf.range(1, tf.shape(points)[0]+1), # from the first point to the last point
            initializer=np.array(0., dtype=np.float32))
        print(sess.run(r))

        # # test sampling
        # seqs = hawkes.sampling(sess, batch_size=10)
        # print(seqs)