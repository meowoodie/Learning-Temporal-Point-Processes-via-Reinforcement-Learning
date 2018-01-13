Imitation-Learning-for-Point-Process
===

Introduction
---
PPG (Point Process Generator) is a highly-customized RNN (Recurrent Neural Network) Model that is able to produce actions by imitating expert sequences.

How to Train a PPG
---
Before training a PPG, you have to organize and format the training data and test data into numpy arrays, which have shape (`num_seqs`, `max_num_actions`, `num_features`) and (`batch_size`, `max_num_actions`, `num_features`) respectively. Also you have to do paddings (zero values) for those sequences of actions whose length are less than `max_num_actions`. For the time being, 
`num_features` has to be set as 1 (time). 

Then you can initiate a session by tensorflow, and do the training process like following example:
```python
with tf.Session() as sess:
	# Substantiate a ppg object
	ppg = PointProcessGenerater(
		seq_len=seq_len,
		batch_size=batch_size, 
		state_size=state_size,
		feature_size=feature_size)
	# Start training
	ppg.train(sess, input_data, test_data, iters=10, display_step=1, pretrained=False)
```
The details of the training process will be logged into standard error stream. 

How to Generate Actions
---
By simply running following code, fixed size (number and length) of sequences with indicated time frame will be generated automatically without input data.
```python
with tf.Session() as sess:

	# Here is the code for training a new ppg or loading an existed ppg

	# Generate actions
	ppg.generate(sess, num_seq, max_t, max_learner_len=10, pretrained=False)
```

> A runnable demo has been attached to this repo in `demo.py`.


