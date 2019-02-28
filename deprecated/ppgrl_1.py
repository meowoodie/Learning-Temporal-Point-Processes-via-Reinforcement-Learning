import sys
import arrow
import utils
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tfgen_1 import SpatialTemporalHawkes

class RL_Hawkes_Generator(object):
    """
    Reinforcement Learning Based Point Process Generator
    """

    def __init__(self, batch_size, lr, keep_latest_k=None, T=[0., 10.], S=[[-1., 1.], [-1., 1.]], C=1., maximum=1e+3):
        """
        Params:
        - T: the maximum time of the sequences
        - S: the space of location
        - C: the constant in diffusion kernel
        """
        # model hyper-parameters
        self.batch_size = batch_size
        self.T          = T # maximum time
        self.S          = S # location space
        # input tensors: expert sequences (time, location)
        self.input_seqs = tf.placeholder(tf.float32, [batch_size, None, 3])
        # Hawkes process generator
        self.hawkes     = SpatialTemporalHawkes(C=C, maximum=maximum)
        # generated tensors: learner sequences (time, location, loglikelihood)
        # - learner_seqs:        [batch_size, seq_len, data_dim]
        # - learner_seqs_loglik: [batch_size, seq_len, 1]
        self.seqs, logliks = self.hawkes.sampling(T, S, batch_size=batch_size, keep_latest_k=None)
        # build policy optimizer
        self._policy_optimizer(
            expert_seqs=self.input_seqs, 
            learner_seqs=self.seqs, learner_seqs_loglik=logliks, 
            lr=lr)
    
    def _policy_optimizer(self, expert_seqs, learner_seqs, learner_seqs_loglik, lr):
        """
        """
        # concatenate batches in the sequences
        concat_expert_seq         = self.__concatenate_batch(expert_seqs)          # [batch_size * expert_seq_len, data_dim]
        concat_learner_seq        = self.__concatenate_batch(learner_seqs)         # [batch_size * learner_seq_len, data_dim]
        concat_learner_seq_loglik = self.__concatenate_batch(learner_seqs_loglik)  # [batch_size * learner_seq_len, 1]

        # calculate average rewards
        print("[%s] building reward." % arrow.now(), file=sys.stderr)
        reward = self._reward(concat_expert_seq, concat_learner_seq) 

        # cost and optimizer
        print("[%s] building optimizer." % arrow.now(), file=sys.stderr)
        self.cost      = tf.reduce_sum(tf.multiply(reward, concat_learner_seq_loglik), axis=0)
        self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.cost)

    def _reward(self, expert_seq, learner_seq, kernel_bandwidth=0.5): 
        """reward function"""
        # get mask for concatenated expert and learner sequences
        learner_mask_t = tf.expand_dims(tf.cast(learner_seq[:, 0] > 0, tf.float32), -1)
        expert_mask_t  = tf.expand_dims(tf.cast(expert_seq[:, 0] > 0, tf.float32), -1)

        # calculate mask for kernel matrix
        learner_learner_kernel_mask = tf.matmul(learner_mask_t, tf.transpose(learner_mask_t))
        expert_learner_kernel_mask  = tf.matmul(expert_mask_t, tf.transpose(learner_mask_t))

        # calculate upper-half kernel matrix
        # - [learner_seq_len, learner_seq_len], [expert_seq_len, learner_seq_len]
        learner_learner_kernel, expert_learner_kernel = self.__kernel_matrix(learner_seq, expert_seq, kernel_bandwidth)

        learner_learner_kernel = tf.multiply(learner_learner_kernel, learner_learner_kernel_mask)
        expert_learner_kernel  = tf.multiply(expert_learner_kernel, expert_learner_kernel_mask)

        # calculate reward for each of data point in learner sequence
        emp_ll_mean = tf.reduce_sum(learner_learner_kernel, axis=0) * 2 # [batch_size * learner_seq_len]
        emp_el_mean = tf.reduce_sum(expert_learner_kernel, axis=0) * 2  # [batch_size * learner_seq_len]
        return tf.expand_dims(emp_ll_mean - emp_el_mean, -1)            # [batch_size * learner_seq_len, 1]
    
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

    def train(self, sess, 
            epoches,               # number of epoches (how many times is the entire dataset going to be trained)
            expert_seqs,           # [n, seq_len, 3]
            trainplot=True,        # plot the change of intensity over epoches
            pretrained=False):
        """Train the point process generator given expert sequences."""

        # initialization
        if not pretrained:
            print("[%s] parameters are initialized." % arrow.now(), file=sys.stderr)
            # initialize network parameters
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

        # data configurations
        # - number of expert sequences
        n_data    = expert_seqs.shape[0]
        # - number of batches
        n_batches = int(n_data / self.batch_size)
        
        if trainplot:
            ppim = utils.PointProcessIntensityMeter(self.T[1], batch_size)

        # training over epoches
        for epoch in range(epoches):
            # shuffle indices of the training samples
            shuffled_ids = np.arange(n_data)
            np.random.shuffle(shuffled_ids)

            # training over batches
            avg_train_cost = []
            for b in range(n_batches):
                idx             = np.arange(self.batch_size * b, self.batch_size * (b + 1))
                # training and testing indices selected in current batch
                batch_train_ids = shuffled_ids[idx]
                # training and testing batch data
                batch_train_expert = expert_seqs[batch_train_ids, :, :]
                # print(sess.run(self.seqs))
                # optimization procedure
                sess.run(self.optimizer, feed_dict={self.input_seqs: batch_train_expert})
                # cost for train batch and test batch
                train_cost = sess.run(self.cost, feed_dict={self.input_seqs: batch_train_expert})
                print("[%s] batch training cost: %.2f." % (arrow.now(), train_cost), file=sys.stderr)
                # record cost for each batch
                avg_train_cost.append(train_cost)

            if trainplot:
                # update intensity plot
                learner_seqs, _ = self.hawkes.get_learner_seqs(sess, self.batch_size, keep_latest_k=None)
                ppim.update_time_intensity(batch_train_expert[:, : , 0], learner_seqs[:, :, 0])
                ppim.update_location_intensity(batch_train_expert[:, : , 1:], learner_seqs[:, :, 1:])

            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % (arrow.now(), epoch, n_batches, self.batch_size), file=sys.stderr)
            print('[%s] Training cost:\t%f' % (arrow.now(), avg_train_cost), file=sys.stderr)