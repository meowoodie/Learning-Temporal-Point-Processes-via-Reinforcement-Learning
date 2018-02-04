import tensorflow as tf
import numpy as np
from imitpp_song import PointProcessGenerator6
# import Simulate_Poisson as SP
import utils.ppgen as SP
import pdb

# p = SP.IntensityHomogenuosPoisson(1.0)
# p = SP.IntensityPoisson(-0.2, 3.5)
# p = SP.IntensityPoisson(0.2, 0.5)
# p = SP.IntensityGaussianMixture(k=2, centers=[2, 4], stds=[1, 1])

T_max = 15
seq_num = 2000

intensity = SP.IntensityHawkesPlusPoly(mu=1, alpha=0, beta=1,
                                    segs=[0, T_max/4., T_max*2./4., T_max*3./4., T_max],
                                    b=0, A=[2, -2, 2, -2])
expert_time = SP.generate_sample(intensity, T=T_max, n=seq_num)

print("here")

# expert_time = SP.generate_sample(p, T_max, seq_num)

batch_size = 32
expert_len = []
for s in expert_time:
    expert_len.append(len(s))
max_expert_len = max(expert_len)
seq_len = max_expert_len + 30

ee = T_max * np.ones((seq_num, seq_len))
for i in range(seq_num):
    ee[i, 0:expert_len[i]] = np.array(expert_time[i])

# pdb.set_trace()

# print("sequence length is:")
# print(seq_len)
# print(ee)

ppg = PointProcessGenerator6(kernel_bandwidth=0.5,
                             T_max=T_max,
                             state_size=32,
                             seq_len=seq_len,
                             batch_size=batch_size)

with tf.Session() as sess:
    ppg.train3(sess,
              expert_time_pool=ee,
              )
