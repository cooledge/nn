import tensorflow as tf
import numpy as np
import pdb

# use rnn
#   dont forget parenthesis

# inputs sequence of operators
# output selected operator

batch_size = 10
seq_len = 8
lstm_size = 128
number_of_layers = 1
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
  
number_of_operators = len(operators)
# 0 is nothing, 1.. are the operators
number_of_chars = number_of_operators + 1

model_inputs = tf.placeholder(tf.int32, shape=(batch_size, seq_len))
model_outputs = tf.placeholder(tf.int32, shape=(batch_size, 1))

model_one_hot_inputs = [tf.squeeze(tf.one_hot(split, number_of_chars)) for split in tf.split(model_inputs, seq_len, axis=1)]
model_one_hot_outputs = tf.one_hot(model_outputs, number_of_operators)

"""
# check one_one_inputs
session = tf.Session()
session.run(tf.global_variables_initializer())
i = [[0, 1, 2, 3, 4]] * batch_size
pdb.set_trace()
ohi = session.run(tf.split(model_inputs, seq_len, axis=1), { model_inputs: i } )
ohi = session.run(tf.one_hot([1,2,3,4,5], seq_len, axis=1))
pdb.set_trace()
ohi = session.run(model_one_hot_inputs, { model_inputs: i } )
"""

model_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
model_multi_cell = tf.contrib.rnn.MultiRNNCell([model_lstm]*number_of_layers)
model_initial_state = model_multi_cell.zero_state(batch_size, tf.float32)

model_rnn_outputs, model_rnn_state = tf.contrib.rnn.static_rnn(model_multi_cell, model_one_hot_inputs, model_initial_state)

# (batch_size, seq_len*lstm_size)
model_rnn_outputs_seq = tf.concat(model_rnn_outputs, 1)

"""
i = [ [[1,1,1], [1,1,1]], [[2,2,2], [2,2,2]], [[3,3,3], [3,3,3]] ]
session.run(tf.concat(i, 1))
"""

# fully connected layer to one hot vector of expected outputs
W = tf.Variable(tf.random_normal([seq_len*lstm_size, number_of_operators]))
b = tf.Variable(tf.zeros([number_of_operators]))

model_logits = tf.matmul(model_rnn_outputs_seq, W) + b
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
    loss, _ = session.run([model_loss, model_train_op], { model_inputs: inputs_train[start:end], model_outputs: outputs_train[start:end] })
    print("\tloss({0})".format(loss))

# check it

inputs_test, outputs_test = load_files("test")

number_of_batches = int(len(inputs_test)/batch_size)

right = 0
wrong = 0
for batch_no in range(number_of_batches):
  start = batch_no * batch_size
  end = start + batch_size
  selected = session.run(tf.argmax(model_probs, axis=1), { model_inputs: inputs_test[start:end] })
  for i in range(batch_size):
    if selected[i] == outputs_test[start+i]:
      right += 1
    else:
      wrong += 1

print("right({0}), wrong({1})".format(right, wrong))

