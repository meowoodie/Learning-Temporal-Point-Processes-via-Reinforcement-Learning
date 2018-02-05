import tensorflow as tf
import numpy as np
from imitpp_song import PointProcessGenerator6
# import Simulate_Poisson as SP
import utils.ppgen as SP
import pdb

# T_max      = 15
# batch_size = 16
# seq_num    = 2000
#
# intensity = SP.IntensityHawkesPlusPoly(mu=0.5, alpha=0, beta=1,
#                                        segs=[0, T_max/2., T_max],
#                                        b=0, A=[1, -1])
# expert_seqs = SP.generate_sample(intensity, T=T_max, n=seq_num)
#
# seq_len = max(map(len, expert_seqs))
#
# # Padding expert sequences in accordance with the sequence length
# ee = T_max * np.ones((seq_num, seq_len))
# for i in range(seq_num):
#     ee[i, 0:len(expert_seqs[i])] = np.array(expert_seqs[i])
#
# ppg = PointProcessGenerator6(kernel_bandwidth=0.5,
#                              T_max=T_max,
#                              state_size=32,
#                              seq_len=seq_len,
#                              batch_size=batch_size)
#
# with tf.Session() as sess:
#     ppg.train3(sess, expert_time_pool=ee)

from glob import glob
fnames = glob("data/iter_*_learner_seqs.txt")
print fnames
seqs_list  = [ np.loadtxt(f, delimiter=",") for f in fnames if f is not ".DS_Store" ]
final_seqs = np.concatenate(seqs_list)
print final_seqs.shape
