#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for testing basic version of imitpp on synthetic dataset
"""

import sys
import arrow
import utils
import random
import numpy as np
import tensorflow as tf

from ppgrl import RL_Hawkes_Generator
from stppg import HawkesLam, GaussianMixtureDiffusionKernel, StdDiffusionKernel

# Avoid error msg [OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.]
# Reference: https://github.com/dmlc/xgboost/issues/1715
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
	data   = np.load('../Spatio-Temporal-Point-Process-Simulator/data/apd.crime.perday.npy')
	params = np.load('../Spatio-Temporal-Point-Process-Simulator/data/gaussian_mixture_params.npz')

	da     = utils.DataAdapter(init_data=data)
	mu     = params['mu']
	# kernel = GaussianMixtureDiffusionKernel(
	# 	n_comp=5, layers=[5], C=1., beta=params['beta'], 
	# 	SIGMA_SHIFT=.05, SIGMA_SCALE=.2, MU_SCALE=.1,
	# 	Wss=params['Wss'], bss=params['bss'], Wphis=params['Wphis'])
	kernel = GaussianMixtureDiffusionKernel(
		n_comp=20, layers=[5], C=1., beta=.8, 
		SIGMA_SHIFT=.05, SIGMA_SCALE=.3, MU_SCALE=.03)
	# kernel = StdDiffusionKernel(C=1., beta=3., sigma_x=.15, sigma_y=.15)
	lam    = HawkesLam(mu, kernel, maximum=1e+3)

	# ngrid should be smaller than 100, due to the computational 
	# time is too large when n > 100.
	utils.spatial_intensity_on_map(
		"test.html", da, lam, data, t=1.0, 
		xlim=[33.70, 33.87],
		ylim=[-84.50, -84.30],
		ngrid=200)
		# xlim=da.xlim, ylim=da.ylim, ngrid=200)
