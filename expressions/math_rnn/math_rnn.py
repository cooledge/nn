import tensorflow as tf
import numpy as np
import pdb

# use rnn
#   dont forget parenthesis

# inputs sequence of operators
# output selected operator

batch_size = 10
seq_len = 5
lstm_size = 128
number_of_layers = 3

number_of_operators = 5

pdb.set_trace()
inputs = tf.placeholder(tf.int32, shape=(batch_size, seq_len))
outputs = tf.placeholder(tf.int32, shape=(batch_size, 1))

one_hot_inputs = [tf.squeeze(tf.one_hot(split, number_of_operators)) for split in tf.split(inputs, seq_len, axis=1)]

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
multi_cell = tf.contrib.rnn.MultiRNNCell([lstm]*number_of_layers)
initial_state = multi_cell.zero_state(batch_size, tf.float32)

outputs = state = tf.contrib.rnn.static_rnn(multi_cell, inputs, initial_state, tf.float32)
