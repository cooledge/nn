# implement char-rnn in tensorflow

import os
import argparse
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq as seq2seq
#from tensorflow.python.ops import seq2seq
import pdb

# implement char_rnn_tf 

parser = argparse.ArgumentParser(description="Program to train a story generator")
parser.add_argument("--data", default="./data", type=str, help="directory with input.txt")
parser.add_argument("--seq_len", default=50, type=int, help="length of the sequences for the rnn")
parser.add_argument("--epochs", default=50, type=int, help="nummer of epichs tu run")
batch_size = 60
batch_len = 50
cell_state_size = 128 # in cell
rnn_cells_depth = 2 # cells in each time step

args = parser.parse_args()
print(args.data)
epochs = args.epochs

model_dir = os.path.dirname(os.path.abspath(__file__)) + "/model"
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
model_filename = model_dir + "/model"

batch_size_in_chars = batch_size*batch_len

# get the input and output data
def load_file(file_dir):
  file = open(os.path.join(file_dir, "input.txt"), "r")
  data = file.read()
  data = data.lower()

  words = data.split()
  # make the input and target

  number_of_letters = len(words)
  word_to_id_map = {}
  id_to_word_map = {}
  for i, words in enumerate(words):
    word_to_id_map[ch] = i
    id_to_ch_map[i] = ch

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

  return([input, targets, number_of_letters, ch_to_id_map, id_to_ch_map])

input, targets, number_of_letters, ch_to_id_map, id_to_ch_map = load_file(args.data)

# setup the model

def model(cell_state_size, rnn_cells_depth, batch_size, batch_len, number_of_letters, reuse): 

  input_placeholder = tf.placeholder(tf.int32, shape=(None, batch_len), name="input")
  target_placeholder = tf.placeholder(tf.int32, shape=(None, batch_len), name="target")
  # make dictionary for letters (60, 128)

  with tf.variable_scope("rnn") as scope:
    if reuse:
      scope.reuse_variables()

    cell = tf.nn.rnn_cell.BasicLSTMCell(cell_state_size)
    rnn_cell= tf.nn.rnn_cell.MultiRNNCell([cell] * rnn_cells_depth)

    W = tf.get_variable("W", shape=(128, number_of_letters))
    b = tf.get_variable("b", shape=(number_of_letters))

    embedding = tf.get_variable("embedding", [number_of_letters, 128])
    # (60, 50, 128)
    rnn_input = tf.nn.embedding_lookup(embedding, input_placeholder)
    # 50 of (60, 1, 128)
    rnn_input = tf.split(rnn_input, batch_len, axis=1)
    rnn_input = [ tf.squeeze(rni, [1]) for rni in rnn_input ]


    # map input from id numbers to rnn states
    decoder_initial_state = rnn_cell.zero_state(batch_size, tf.float32)
    # outputs list of 50 - (60,128)
    outputs, last_state = seq2seq.rnn_decoder(rnn_input, decoder_initial_state, rnn_cell, scope="rnn")
  # (60, -1)
  outputs = tf.concat(outputs, 1)
  # (-1, 128) ie a list of letters
  outputs = tf.reshape(outputs, [-1, 128])


  # (3000, number_of_letters) 
  logits = tf.matmul(outputs, W) + b
  #probs = tf.nn.softmax(logits, 1, name="probs")
  probs = tf.nn.softmax(logits, -1, name="probs")

  loss = seq2seq.sequence_loss_by_example([logits], [tf.reshape(target_placeholder, [-1])], [tf.ones([batch_size * batch_len])], number_of_letters)
  return([loss, probs, decoder_initial_state, input_placeholder, target_placeholder, last_state, logits])

def optimizer(batch_size, batch_len, loss):
  lr = tf.Variable(0.0, trainable=False)
  tvars = tf.trainable_variables()

  optimizer = tf.train.AdamOptimizer(lr)
  cost_op = tf.reduce_sum(loss) / batch_size / batch_len
  grads= tf.gradients(cost_op, tvars)
  grad_clip = 5
  tf.clip_by_global_norm(grads, grad_clip)
  grads_and_vars = zip(grads, tvars)
  train_op = optimizer.apply_gradients(grads_and_vars)

  return [train_op, cost_op, lr]

loss, probs, decoder_initial_state, input_placeholder, target_placeholder, last_state, logits = model(cell_state_size, rnn_cells_depth, batch_size, batch_len, number_of_letters, False)
train_op, cost_op, lr = optimizer(batch_size, batch_len, loss)

# train the model

session = tf.Session()
session.run(tf.global_variables_initializer())

saver = tf.train.Saver()

try:
  saver.restore(session, model_filename)
except Exception as e:
  0 # ignore 

learning_rate = 0.002
decay = 0.97
for epoch in range(epochs):
  print("Epoch %s" %(epoch+1))
  session.run(tf.assign(lr, learning_rate * (decay ** epoch)))
  state = session.run(decoder_initial_state)
  for i, t in zip(input, targets):
    feed = {input_placeholder: i, target_placeholder: t}
    for i, (c, h) in enumerate(decoder_initial_state):
      feed[c] = state[i].c
      feed[h] = state[i].h
    cost, _, state = session.run([cost_op, train_op, last_state], feed_dict=feed)

saver.save(session, model_filename)

# sample the model

loss, probs, decoder_initial_state, input_placeholder, target_placeholder, last_state, logits = model(cell_state_size, rnn_cells_depth, 1, 1, number_of_letters, True)

def predict_words(prime):
  #prime = "MEN"
  result = prime
  state = session.run(decoder_initial_state)
  for ch in prime:
    id = ch_to_id_map[ch]
    i = [[id]]
    t = [[id]]
     
    feed = {input_placeholder: i, decoder_initial_state: state}
    #for i, (c, h) in enumerate(decoder_initial_state):
      #feed[c] = state[i].c
      #feed[h] = state[i].h

    actual_probs, state = session.run([probs, last_state], feed_dict=feed)

  # now the super hacky part. Generate one character at a time. If we generate
  # exactly what the RNN says the output is very repetitive so instead of doing
  # that we get the probabilities for each output and randomly select a character
  # based on its likelyhood. That makes the output not boring. 
  # FUTURE WORK: get the hacky bit in the neural network

  def hacky_character_picker( probs ):
    cs = np.cumsum(probs)
    t = np.sum(probs)
    r = np.random.rand(1)
    cutoff = r * t
    return int(np.searchsorted(cs, cutoff))
   
  for i in range(500):
    id = hacky_character_picker(actual_probs)
    #id = np.argmax(actual_probs)
    ch = id_to_ch_map[id]
    result = result + ch
    i = [[id]]
    
    feed = {input_placeholder: i, decoder_initial_state: state}
    #feed = {input_placeholder: i}
    #for i, (c, h) in enumerate(decoder_initial_state):
      #feed[c] = state[i].c
      #feed[h] = state[i].h

    actual_probs, state = session.run([probs, last_state], feed_dict=feed)

print("AND the result is: ")
print(predict_words("men"))



