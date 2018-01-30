import tensorflow as tf
import numpy as np

from imitpp_v2 import PointProcessGenerator5
from utils.ppgen import *

T_max = 15.
seq_num = 2000
batch_size = 64

intensity = IntensityHawkesPlusPoly(mu=1, alpha=0.3, beta=1,
                                    segs=[0, T_max/4., T_max*2./4., T_max*3./4., T_max],
                                    b=1, A=[1, -1, 1, -1])

expert_time = generate_sample(intensity, T=T_max, n=seq_num)

expert_len = []
for s in expert_time:
    expert_len.append(len(s))
max_expert_len = max(expert_len)
seq_len = max_expert_len + 20

ee = T_max * np.ones((seq_num, seq_len))
for i in range(seq_num):
    ee[i, 0:expert_len[i]] = np.array(expert_time[i])

ppg = PointProcessGenerator5(kernel_bandwidth=0.5,
                             T_max=T_max,
                             state_size=16,
                             seq_len=seq_len,
                             batch_size=batch_size)

with tf.Session() as sess:
    ppg.train3(sess, expert_time_pool=ee)
    # tf_saver = tf.train.Saver()
    # tf_saver.save(sess, "resource/model.song_le/imitpp5")

    iters = 50
# with tf.Session() as sess:
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # tf_saver = tf.train.Saver()
    # tf_saver.restore(sess, "resource/model.song_le/imitpp5")
    learner_seqs = np.zeros((0, seq_len))
    for ind in range(iters):
        rand_uniform_pool = np.random.rand(batch_size, seq_len).astype(np.float32)
        output_seqs, cum_output_seqs = sess.run(ppg.generate3(rand_uniform_pool))
        learner_seqs = np.concatenate((learner_seqs, cum_output_seqs), axis=0)
    # print learner_seqs.shape
    np.savetxt("resource/generation/learner_seq.txt", learner_seqs, delimiter=',')
