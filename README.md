Imitation-Learning-for-Point-Process
===

Introduction
---
PPG (Point Process Generator) is a highly-customized RNN (Recurrent Neural Network) Model that would be able to produce actions by imitating expert sequences. (**Shuang Li's ongoing work**)

How to Train a PPG
---
Before training a PPG, you have to organize and format the training data and test data into numpy arrays, which have shape (`num_seqs`, `max_num_actions`, `num_features`) and (`batch_size`, `max_num_actions`, `num_features`) respectively. Also you have to do paddings (zero values) for those sequences of actions whose length are less than `max_num_actions`. For the time being,
`num_features` has to be set as 1 (time).

Then you can initiate a session by tensorflow, and do the training process like following example:
```python
max_t        = 7
seq_len      = 10
batch_size   = 3
state_size   = 5
feature_size = 1
with tf.Session() as sess:
	# Substantiate a ppg object
	ppg = PointProcessGenerator(
		max_t=max_t, # max time for all learner & expert actions
		seq_len=seq_len, # length for all learner & expert actions sequences
		batch_size=batch_size,
		state_size=state_size,
		feature_size=feature_size,
		iters=10, display_step=1, lr=1e-4)
	# Start training
	ppg.train(sess, input_data, test_data, pretrained=False)
```
You can also omit parameter `test_data`, which is set `None` by default, if you don't have test data for training.

The details of the training process will be logged into standard error stream. Below is testing log information.
```shell
2018-01-12 21:58:19.597947: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-12 21:58:19.597973: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-01-12 21:58:19.597979: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-12 21:58:19.597984: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
[2018-01-12T21:58:23.173787-05:00] Iter: 2
[2018-01-12T21:58:23.173927-05:00] Train Loss: -6.65403,	Test Loss: -1.71365
[2018-01-12T21:58:23.214679-05:00] Iter: 4
[2018-01-12T21:58:23.214815-05:00] Train Loss: -5.15812,	Test Loss: -6.43803
[2018-01-12T21:58:23.267725-05:00] Iter: 6
[2018-01-12T21:58:23.267862-05:00] Train Loss: -4.56620,	Test Loss: -8.62218
[2018-01-12T21:58:23.316168-05:00] Iter: 8
[2018-01-12T21:58:23.316304-05:00] Train Loss: -3.63575,	Test Loss: -6.72692
[2018-01-12T21:58:23.373238-05:00] Iter: 10
[2018-01-12T21:58:23.373373-05:00] Train Loss: -9.03363,	Test Loss: -4.45932
[2018-01-12T21:58:23.373500-05:00] Optimization Finished!
```

How to Generate Actions
---
By simply running following code, fixed size (number and length) of sequences with indicated time frame will be generated automatically without input data. What needs to be noted is the length and the number of the generated sequence have been specified by the same input parameters when you initialize the `ppg` object.
```python
with tf.Session() as sess:

	# Here is the code for training a new ppg or loading an existed ppg

	# Generate actions
	actions, states_history = ppg.generate(sess, pretrained=False)
	print actions
```
Below are generated test actions.
```shell
(array([[ 0.63660634,  1.12912512,  0.39286253],
        [ 1.64375508,  1.60563707,  1.77609217],
        [ 3.08153439,  2.41127753,  2.59949875],
        [ 3.91807413,  3.74258327,  3.54215193],
        [ 4.97372961,  4.49850368,  4.98060131],
        [ 5.73539734,  5.15121365,  5.43891001],
        [ 6.24749708,  5.667624  ,  6.38705158],
        [ 6.60757065,  6.88907528,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]], dtype=float32)
```


> A runnable demo has been attached to this repo in `demo.py`.
