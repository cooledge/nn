# implement char-rnn in tensorflow

import os
import argparse
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
import pdb

# implement char_rnn_tf 

parser = argparse.ArgumentParser(description="Program to train a story generator")
parser.add_argument("--data", default="./data", type=str, help="directory with input.txt")
parser.add_argument("--seq_len", default=50, type=int, help="length of the sequences for the rnn")
epochs = 1
batch_size = 60
batch_len = 50
cell_state_size = 128 # in cell
rnn_cells_depth = 2 # cells in each time step

args = parser.parse_args()
print(args.data)

batch_size_in_chars = batch_size*batch_len

# get the input and output data
def load_file(file_dir):
  file = open(os.path.join(file_dir, "input.txt"), "r")
  data = file.read()

  # make the input and target

  counter = collections.Counter(data)
  number_of_letters = len(counter.keys())
  ch_to_id_map = {}
  for i, ch in enumerate(counter.keys()):
    ch_to_id_map[ch] = i

  input = []

  for ch in data:
    input.append(ch_to_id_map[ch])
  input = np.array(input)
  targets = input.copy()
  targets[:-1] = input[1:]
  targets[-1] = input[0]


  number_of_batches = len(input)//batch_size_in_chars

  def setup_inputs(input):
    input = input[0:(number_of_batches*batch_size_in_chars)]
    input = np.reshape(input, [batch_size, -1])
    input = np.split(input, number_of_batches, 1)
    return input

  # (60, 50)
  input = setup_inputs(input)
  # (60, 50)
  targets = setup_inputs(targets)

  return([input, targets, number_of_letters, ch_to_id_map])

input, targets, number_of_letters, ch_to_id_map = load_file(args.data)

# setup the model

def model(cell_state_size, rnn_cells_depth, batch_size, batch_len, number_of_letters, reuse): 

  cell = tf.nn.rnn_cell.BasicLSTMCell(cell_state_size)
  rnn_cell= tf.nn.rnn_cell.MultiRNNCell([cell] * rnn_cells_depth)
  input_placeholder = tf.placeholder(tf.int32, shape=(batch_size, batch_len), name="input")
  target_placeholder = tf.placeholder(tf.int32, shape=(batch_size, batch_len), name="target")
  # make dictionary for letters (60, 128)

  with tf.variable_scope("rnn") as scope:
    if reuse:
      scope.reuse_variables()

    W = tf.get_variable("W", shape=(128, number_of_letters))
    b = tf.get_variable("b", shape=(number_of_letters))

    embedding = tf.get_variable("embedding", [number_of_letters, 128])
    # (60, 50, 128)
    rnn_input = tf.nn.embedding_lookup(embedding, input_placeholder)
    # 50 of (60, 1, 128)
    rnn_input = tf.split(1, batch_len, rnn_input)
    rnn_input = [ tf.squeeze(rni, [1]) for rni in rnn_input ]


    # map input from id numbers to rnn states
    decoder_initial_state = rnn_cell.zero_state(batch_size, tf.float32)
    # outputs list of 50 - (60,128)
    outputs, last_state = seq2seq.rnn_decoder(rnn_input, decoder_initial_state, rnn_cell, scope="rnn")
  # (60, -1)
  outputs = tf.concat(1, outputs)
  # (-1, 128) ie a list of letters
  outputs = tf.reshape(outputs, [-1, 128])


  # (3000, number_of_letters) 
  logits = tf.matmul(outputs, W) + b
  #probs = tf.nn.softmax(logits, 1, name="probs")
  probs = tf.nn.softmax(logits, -1, name="probs")

  loss = seq2seq.sequence_loss_by_example([logits], [tf.reshape(target_placeholder, [-1])], [tf.ones([batch_size * batch_len])], number_of_letters)
  lr = tf.Variable(1.0, trainable=False)
  tvars = tf.trainable_variables()

  optimizer = tf.train.AdamOptimizer(lr)
  cost_op = tf.reduce_sum(loss) / batch_size / batch_len
  grads= tf.gradients(cost_op, tvars)
  grad_clip = 5
  tf.clip_by_global_norm(grads, grad_clip)
  grads_and_vars = zip(grads, tvars)
  train_op = optimizer.apply_gradients(grads_and_vars)

  return([train_op, probs, decoder_initial_state, input_placeholder, target_placeholder, cost_op, last_state, logits])

train_op, probs, decoder_initial_state, input_placeholder, target_placeholder, cost_op, last_state, logits = model(cell_state_size, rnn_cells_depth, batch_size, batch_len, number_of_letters, False)

# train the model

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess
for epoch in range(epochs):
  print("Epoch %s" %(epoch+1))
  state = sess.run(decoder_initial_state)
  for i, t in zip(input, targets):
    #pdb.set_trace()
    feed = {input_placeholder: i, target_placeholder: t}
    for i, (c, h) in enumerate(decoder_initial_state):
      feed[c] = state[i].c
      feed[h] = state[i].h
    cost, _, state = sess.run([cost_op, train_op, last_state], feed_dict=feed)
    #print("cost %i" % (cost))

# sample the model

train_op, probs, decoder_initial_state, input_placeholder, target_placeholder, cost_op, last_state, logits = model(cell_state_size, rnn_cells_depth, 1, 1, number_of_letters, True)

prime = "The "
state = sess.run(decoder_initial_state)
for ch in prime:
  id = ch_to_id_map[ch]
  i = [[id]]
  t = [[id]]
   
  feed = {input_placeholder: i, target_placeholder: t}
  for i, (c, h) in enumerate(decoder_initial_state):
    feed[c] = state[i].c
    feed[h] = state[i].h

  actual_probs, state = sess.run([probs, last_state], feed_dict=feed)




