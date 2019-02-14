import sys
import arrow
import utils
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from mstlstm import MarkedSpatialTemporalLSTM

class RLPointProcessGenerator(object):
    """
    Reinforcement Learning Based Point Process Generator
    """

    def __init__(self, T, seq_len, lstm_hidden_size, loc_hidden_size, mak_hidden_size, m_dim):
        """
        Params:
        - T:                the maximum time of the sequences
        - seq_len:          the length of the sequences
        - lstm_hidden_size: size of hidden state of the LSTM
        - loc_hidden_size:  size of hidden feature of location
        - mak_hidden_size:  size of hidden feature of mark
        - m_dim:            number of categories of marks
        """
        # model hyper-parameters
        self.T       = T                # maximum time
        self.t_dim   = 1                # by default
        self.l_dim   = 2                # by default
        self.m_dim   = m_dim            # number of categories of marks
        self.seq_len = seq_len          # length of each generated sequences
        # LSTM generator
        self.mstlstm  = MarkedSpatialTemporalLSTM(
            step_size=seq_len, lstm_hidden_size=lstm_hidden_size, 
            loc_hidden_size=loc_hidden_size, mak_hidden_size=mak_hidden_size, m_dim=m_dim)
    
    def _initialize_policy_network(self, sess, batch_size, starter_learning_rate=0.01, decay_rate=0.99, decay_step=100):
        """
        Construct Policy Network
        
        Policy should be flexible and expressive enough to capture the potential complex point process patterns of data.
        Therefore, a customized recurrent neural network (RNN) with stochastic neurons is adopted, where hidden state is 
        computed by hidden state of last moment and stochastically generated action. i.e.
          a_{i+1} is sampling from pi(a|h_{i})
          h_{i+1} = rnn_cell(h_{i}, a_{i+1})
        """
        # input tensors: expert sequences (time, location, marks)
        self.input_seq_t = tf.placeholder(tf.float32, [batch_size, None, self.t_dim])
        self.input_seq_l = tf.placeholder(tf.float32, [batch_size, None, self.l_dim])
        self.input_seq_m = tf.placeholder(tf.float32, [batch_size, None, self.m_dim])

        # construct customized stochastic LSTM network
        self.mstlstm.initialize_network(batch_size)
        # generated tensors: learner sequences (time, location, marks)
        learner_seq_t, learner_seq_l, learner_seq_m = self.mstlstm.seq_t, self.mstlstm.seq_l, self.mstlstm.seq_m
        # log likelihood
        learner_seq_loglik = self.mstlstm.seq_loglik
        # getting training time window (t_0 = 0, T = self.T by default)
        t0, T = 0, self.T # self._training_time_window(learner_seq_t)

        # concatenate batches in the sequences
        expert_seq_t,  expert_seq_l,  expert_seq_m  = \
            self.__concatenate_batch(self.input_seq_t), \
            self.__concatenate_batch(self.input_seq_l), \
            self.__concatenate_batch(self.input_seq_m)
        learner_seq_t, learner_seq_l, learner_seq_m, learner_seq_loglik = \
            self.__concatenate_batch(learner_seq_t), \
            self.__concatenate_batch(learner_seq_l), \
            self.__concatenate_batch(learner_seq_m), \
            self.__concatenate_batch(learner_seq_loglik)
        
        # calculate average rewards
        reward = self._reward(batch_size, t0, T,\
                              expert_seq_t,  expert_seq_l,  expert_seq_m, \
                              learner_seq_t, learner_seq_l, learner_seq_m) # [batch_size*seq_len, 1]

        # cost and optimizer
        self.cost      = tf.reduce_sum(tf.multiply(reward, learner_seq_loglik), axis=0) / batch_size
        global_step    = tf.Variable(0, trainable=False)
        learning_rate  = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.6, beta2=0.9).minimize(self.cost, global_step=global_step)

    def _training_time_window(self, learner_seq_t):
        """
        Time window for the purpose of training. The model only fits a specific segment of the expert sequence
        indicated by 'training_time_window'. This function will return the start time (t_0) and end time (T) of 
        the segment.

        Policy 1:
        t_0 = 0; T = mean(max(learner_seq_t, axis=0))
        """
        # remove invalid time
        mask_t        = self.__get_mask_truncate_by_T(learner_seq_t, self.T) # [batch_size, seq_len, 1]
        learner_seq_t = tf.multiply(learner_seq_t, mask_t)                   # [batch_size, seq_len, 1]
        # policy 1
        t_0 = 0
        T   = tf.reduce_mean(tf.reduce_max(learner_seq_t, axis=0))
        return t_0, T

    def _reward(self, batch_size, t0, T, 
            expert_seq_t, expert_seq_l, expert_seq_m,    # expert sequences
            learner_seq_t, learner_seq_l, learner_seq_m, # learner sequences
            kernel_bandwidth=0.5): 
        """reward function"""
        # get mask for concatenated expert and learner sequences
        expert_seq_mask  = self.__get_mask_truncate_by_T(expert_seq_t, T, t0)  # [batch_size*seq_len, 1]
        learner_seq_mask = self.__get_mask_truncate_by_T(learner_seq_t, T, t0) # [batch_size*seq_len, 1]
        # calculate mask for kernel matrix
        learner_learner_kernel_mask = tf.matmul(learner_seq_mask, tf.transpose(learner_seq_mask))
        expert_learner_kernel_mask  = tf.matmul(expert_seq_mask, tf.transpose(learner_seq_mask))
        # concatenate each data dimension for both expert sequence and learner sequence
        # TODO: Add mark to the sequences
        # expert_seq  = tf.concat([expert_seq_t, expert_seq_l], axis=1)   # [batch_size*seq_len, t_dim+l_dim+m_dim]
        # learner_seq = tf.concat([learner_seq_t, learner_seq_l], axis=1) # [batch_size*seq_len, t_dim+l_dim+m_dim]
        expert_seq  = tf.concat([expert_seq_l], axis=1)                          # [batch_size*seq_len, t_dim]
        learner_seq = tf.concat([learner_seq_l], axis=1)                         # [batch_size*seq_len, t_dim]
        # calculate upper-half kernel matrix
        learner_learner_kernel, expert_learner_kernel = self.__kernel_matrix(
            learner_seq, expert_seq, kernel_bandwidth)                           # 2 * [batch_size*seq_len, batch_size*seq_len]
        learner_learner_kernel = tf.multiply(learner_learner_kernel, learner_learner_kernel_mask)
        expert_learner_kernel  = tf.multiply(expert_learner_kernel, expert_learner_kernel_mask)
        # calculate reward for each of data point in learner sequence
        emp_ll_mean = tf.reduce_sum(learner_learner_kernel, axis=0) * 2 # batch_size*seq_len
        emp_el_mean = tf.reduce_sum(expert_learner_kernel, axis=0) * 2  # batch_size*seq_len
        return tf.expand_dims(emp_ll_mean - emp_el_mean, -1)            # [batch_size*seq_len, 1]

    @staticmethod
    def __get_mask_truncate_by_T(seq_t, T, t_0=0):
        """Masking time, location and mark sequences for the entries before the maximum time T."""
        # get basic mask where 0 if t > T else 1
        mask_t = tf.multiply(
            tf.cast(seq_t < T, tf.float32),
            tf.cast(seq_t > t_0, tf.float32))
        return mask_t # [batch_size*seq_len, 1] or [batch_size, seq_len, 1]
    
    @staticmethod
    def __concatenate_batch(seqs):
        """Concatenate each batch of the sequences into a single sequence."""
        array_seq = tf.unstack(seqs, axis=0)     # [batch_size, seq_len, data_dim]
        seq       = tf.concat(array_seq, axis=0) # [batch_size*seq_len, data_dim]
        return seq
 
    @staticmethod
    def __kernel_matrix(learner_seq, expert_seq, kernel_bandwidth):
        """
        Construct kernel matrix based on learn sequence and expert sequence, each entry of the matrix 
        is the distance between two data points in learner_seq or expert_seq. return two matrix, left_mat 
        is the distances between learn sequence and learn sequence, right_mat is the distances between 
        learn sequence and expert sequence.
        """
        # calculate l2 distances
        learner_learner_mat = utils.l2_norm(learner_seq, learner_seq) # [batch_size*seq_len, batch_size*seq_len]
        expert_learner_mat  = utils.l2_norm(expert_seq, learner_seq)  # [batch_size*seq_len, batch_size*seq_len]
        # exponential kernel
        learner_learner_mat = tf.exp(-learner_learner_mat / kernel_bandwidth)
        expert_learner_mat  = tf.exp(-expert_learner_mat / kernel_bandwidth)
        return learner_learner_mat, expert_learner_mat

    def train(self, sess, batch_size, 
            epoches,               # number of epoches (how many times is the entire dataset going to be trained)
            expert_seq_t,          # [n, seq_len, 1]
            expert_seq_l,          # [n, seq_len, 2]
            expert_seq_m,          # [n, seq_len, m_dim]
            train_test_ratio = 9., # n_train / n_test
            trainplot=True,        # plot the change of intensity over epoches
            pretrained=False):
        """Train the point process generator given expert sequences."""
        # check the consistency of the shape of the expert sequences
        assert expert_seq_t.shape[:-1] == expert_seq_l.shape[:-1] == expert_seq_m.shape[:-1], \
            "inconsistant 'number of sequences' or 'sequence length' of input expert sequences"

        # initialization
        if not pretrained:
            # initialize network parameters
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            # initialize policy network
            self._initialize_policy_network(sess, batch_size)
            

        # data configurations
        # - number of expert sequences
        n_data  = expert_seq_t.shape[0]
        n_train = int(n_data * train_test_ratio / (train_test_ratio + 1.))
        n_test  = int(n_data * 1. / (train_test_ratio + 1.))
        # - number of batches
        n_batches = int(n_train / batch_size)
        # - check if test data size is large enough (> batch_size)
        assert n_test >= batch_size, "test data size %d is less than batch size %d." % (n_test, batch_size)
        
        if trainplot:
            ppim = utils.PointProcessIntensityMeter(self.T, batch_size)

        # training over epoches
        for epoch in range(epoches):
            # shuffle indices of the training samples
            shuffled_ids = np.arange(n_data)
            np.random.shuffle(shuffled_ids)
            shuffled_train_ids = shuffled_ids[:n_train]
            shuffled_test_ids  = shuffled_ids[-n_test:]

            # training over batches
            avg_train_cost = []
            avg_test_cost  = []
            for b in range(n_batches):
                idx             = np.arange(batch_size * b, batch_size * (b + 1))
                # training and testing indices selected in current batch
                batch_train_ids = shuffled_train_ids[idx]
                batch_test_ids  = shuffled_test_ids[:batch_size]
                # training and testing batch data
                batch_train_expert_t = expert_seq_t[batch_train_ids, :, :]
                batch_train_expert_l = expert_seq_l[batch_train_ids, :, :]
                batch_train_expert_m = expert_seq_m[batch_train_ids, :, :]
                batch_test_expert_t  = expert_seq_t[batch_test_ids, :, :]
                batch_test_expert_l  = expert_seq_l[batch_test_ids, :, :]
                batch_test_expert_m  = expert_seq_m[batch_test_ids, :, :]
                # # Debug
                # debug1, debug2 = sess.run([self.mstlstm.test1, self.mstlstm.test2], feed_dict={
                #     self.input_seq_t: batch_test_expert_t,
                #     self.input_seq_l: batch_test_expert_l,
                #     self.input_seq_m: batch_test_expert_m})
                # print(debug1)
                # print(debug2)
                # optimization procedure
                sess.run(self.optimizer, feed_dict={
                    self.input_seq_t: batch_train_expert_t,
                    self.input_seq_l: batch_train_expert_l,
                    self.input_seq_m: batch_train_expert_m})
                # cost for train batch and test batch
                train_cost = sess.run(self.cost, feed_dict={
                    self.input_seq_t: batch_train_expert_t,
                    self.input_seq_l: batch_train_expert_l,
                    self.input_seq_m: batch_train_expert_m})
                test_cost  = sess.run(self.cost, feed_dict={
                    self.input_seq_t: batch_test_expert_t,
                    self.input_seq_l: batch_test_expert_l,
                    self.input_seq_m: batch_test_expert_m})
                # record cost for each batch
                avg_train_cost.append(train_cost)
                avg_test_cost.append(test_cost)

            if trainplot:
                # update intensity plot
                learner_seq_t, learner_seq_l = sess.run(
                    [self.mstlstm.seq_t, self.mstlstm.seq_l], 
                    feed_dict={
                        self.input_seq_t: batch_test_expert_t,
                        self.input_seq_l: batch_test_expert_l,
                        self.input_seq_m: batch_test_expert_m})
                ppim.update_time_intensity(batch_train_expert_t, learner_seq_t)
                ppim.update_location_intensity(batch_train_expert_l, learner_seq_l)

            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            avg_test_cost  = np.mean(avg_test_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % (arrow.now(), epoch, n_batches, batch_size), file=sys.stderr)
            print('[%s] Training cost:\t%f' % (arrow.now(), avg_train_cost), file=sys.stderr)
            print('[%s] Testing cost:\t%f' % (arrow.now(), avg_test_cost), file=sys.stderr)