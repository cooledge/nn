import tensorflow as tf
import argparse
import numpy as np
import pdb

# use rnn
#   dont forget parenthesis

# inputs sequence of operators
# output selected operator

parser = argparse.ArgumentParser(description="Generate the training and test data")
parser.add_argument("--lstm_size", default="128", type=str, help="Size of the LSTM")
parser.add_argument("--lstm_layers", default="1", type=str, help="Number of layers of LSTM's")
parser.add_argument("--l1_size", default="0", type=str, help="Number of layers of LSTM's")
parser.add_argument("--bidi", action='store_true')
args = parser.parse_args()

bidi = args.bidi
l1_size = int(args.l1_size)
batch_size = 10
seq_len = 8
lstm_size = int(args.lstm_size)
number_of_layers = int(args.lstm_layers)
epochs = 20

operators = "+-*/^"
def char_to_id(char):
  return operators.index(char)
def id_to_char(id):
  return operators[id]

def chars_to_ids(chars):
  ids = [0]*seq_len
  for idx, char in enumerate(chars):
    ids[idx] = char_to_id(char)
  return(ids)
def ids_to_chars(chars):
  return([id_to_char(id) for id in ids])

#def input_to_one_hot(ch, input, one_hot):
  #one_hot[ch][input[ch]] = 1

# inputs: (batch_size, seq_len)
# returns: (seq_len, batch_size, number_of_chars)
def inputs_to_one_hots(inputs):
  one_hots = np.zeros((seq_len, batch_size, number_of_chars))
  for b in range(batch_size):
    for ch in range(seq_len):
      one_hots[ch][b][inputs[b][ch]] = 1
  return one_hots  

number_of_operators = len(operators)
# 0 is nothing, 1.. are the operators
number_of_chars = number_of_operators + 1

model_inputs = tf.placeholder(tf.int32, shape=(seq_len, batch_size, number_of_chars))
model_outputs = tf.placeholder(tf.int32, shape=(batch_size, 1))

model_one_hot_inputs = [tf.squeeze(tf.cast(split, tf.float32)) for split in tf.split(model_inputs, seq_len, axis=0)]
model_one_hot_outputs = tf.one_hot(model_outputs, number_of_operators)

model_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
model_multi_cell = tf.contrib.rnn.MultiRNNCell([model_lstm]*number_of_layers)
model_initial_state = model_multi_cell.zero_state(batch_size, tf.float32)

if bidi:
  model_rnn_outputs, model_rnn_state_fw, model_rnn_state_bw = tf.contrib.rnn.static_bidirectional_rnn(model_multi_cell, model_multi_cell, model_one_hot_inputs, model_initial_state, model_initial_state)
else:
  model_rnn_outputs, model_rnn_state = tf.contrib.rnn.static_rnn(model_multi_cell, model_one_hot_inputs, model_initial_state)

# (batch_size, seq_len*lstm_size)
model_rnn_outputs_seq = tf.concat(model_rnn_outputs, 1)

# try another layer with non-linearity
if l1_size > 0:
  W1 = tf.Variable(tf.random_normal([seq_len*lstm_size, l1_size]))
  b1 = tf.Variable(tf.zeros([l1_size]))
  model_logits = tf.matmul(tf.nn.relu(model_rnn_outputs_seq), W1) + b1
else:
  model_logits = model_rnn_outputs_seq
  l1_size = seq_len*lstm_size

if bidi:
  l1_size *= 2

# fully connected layer to one hot vector of expected outputs
W = tf.Variable(tf.random_normal([l1_size, number_of_operators]))
b = tf.Variable(tf.zeros([number_of_operators]))

model_logits = tf.matmul(model_logits, W) + b

model_probs = tf.nn.softmax(model_logits)
model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_one_hot_outputs, logits=model_logits))
model_opt = tf.train.AdamOptimizer()
model_train_op = model_opt.minimize(model_loss)

def load_files(suffix):
  with open("input_{0}.txt".format(suffix)) as file:
    inputs_train = file.read().splitlines()

  with open("output_{0}.txt".format(suffix)) as file:
    outputs_train = file.read().splitlines()

  inputs_train = [ chars_to_ids(chars) for chars in inputs_train ]
  outputs_train = [ [char_to_id(char)] for char in outputs_train ]

  return [inputs_train, outputs_train]

inputs_train, outputs_train = load_files("train")

session = tf.Session()
session.run(tf.global_variables_initializer())

number_of_batches = int(len(inputs_train)/batch_size)
for epoch in range(epochs):
  print("epoch({0})".format(epoch))
  for batch_no in range(number_of_batches):
    start = batch_no * batch_size 
    end = start + batch_size
    one_hots = inputs_to_one_hots(inputs_train[start:end])
    #loss, _, one_hots = session.run([model_loss, model_train_op, model_one_hot_inputs], { model_inputs: inputs_train[start:end], model_outputs: outputs_train[start:end] })
    loss, _ = session.run([model_loss, model_train_op], { model_inputs: one_hots, model_outputs: outputs_train[start:end] })
    print("\tloss({0})".format(loss))

# check it

inputs_test, outputs_test = load_files("test")

number_of_batches = int(len(inputs_test)/batch_size)

if bidi:
  print("Using bidi")

right = 0
wrong = 0
for batch_no in range(number_of_batches):
  start = batch_no * batch_size
  end = start + batch_size
  one_hots = inputs_to_one_hots(inputs_test[start:end])
  selected = session.run(tf.argmax(model_probs, axis=1), { model_inputs: one_hots })
  for i in range(batch_size):
    if selected[i] == outputs_test[start+i]:
      right += 1
    else:
      wrong += 1

print("right({0}), wrong({1})".format(right, wrong))

