# described is at http://gregmcclement.blogspot.ca/2017/03/parsing-natural-languages.html

import tensorflow as tf
import numpy as np
import pdb

# LOADING THE OPERATORS FILE

seq_len = 8
operators = "+-*/^"
number_of_ops = len(operators)

operators = "+-*/^"

# hmmm, this did one run for train 1,2,3 test 4 where the results were perfect

def char_to_id(char):
  return operators.index(char)
def id_to_char(id):
  return operators[id]

def chars_to_ids(chars):
  ids = [None]*seq_len
  for idx, char in enumerate(chars):
    ids[idx] = char_to_id(char)
  return(ids)
def ids_to_chars(chars):
  return([id_to_char(id) for id in ids])

priorities = [
  ('+', '*'),
  ('+', '/'),
  ('-', '*'),
  ('-', '/'),
  ('*', '^'),
  ('/', '^'),
]
batch_size = len(priorities)

inputs_train_pairwise = np.zeros((batch_size, number_of_ops))
outputs_train_pairwise = np.zeros((batch_size, number_of_ops))
batch_no = -1
for k, v in priorities:
  batch_no = batch_no + 1
  outputs_train_pairwise[batch_no][char_to_id(k)] = 1.0
  inputs_train_pairwise[batch_no][char_to_id(v)] = 1.0

def load_files(suffix):
  with open("input_{0}.txt".format(suffix)) as file:
    inputs_train = file.read().splitlines()

  with open("output_{0}.txt".format(suffix)) as file:
    outputs_train = file.read().splitlines()

  # map outputs to the index of the operator to process
  outputs_train = [ inputs_train[i].index(char) for i, char in enumerate(outputs_train) ]
  
  inputs_train = [ chars_to_ids(chars) for chars in inputs_train ]
  #outputs_train = [ [char_to_id(char)] for char in outputs_train ]

  return [inputs_train, outputs_train]

inputs_train, outputs_train = load_files("train")

# END LOADING THE OPERATORS FILE

model_inputs = tf.placeholder(tf.float32, shape=(None, number_of_ops), name="inputs")
# the select op
model_outputs = tf.placeholder(tf.float32, shape=(seq_len), name="outputs")

model_w = tf.get_variable("w", shape=(number_of_ops, number_of_ops), dtype=tf.float32)
model_b = tf.get_variable("b", shape=(number_of_ops), dtype=tf.float32)
model_logits =  tf.matmul(model_inputs, model_w) + model_b
model_predict = tf.nn.softmax(model_logits)

model_combined = tf.squeeze(tf.reshape(tf.reduce_sum(tf.split(model_predict, number_of_ops, axis=1), 1), (1, -1)))
model_reduced = tf.nn.relu(model_inputs - model_combined)
model_op_idx = tf.arg_max(tf.reduce_sum(model_reduced, 1), 0)

model_loss = tf.losses.softmax_cross_entropy(model_outputs, tf.reduce_sum(model_reduced, 1))
model_optimizer = tf.train.AdamOptimizer(0.01)
model_train = model_optimizer.minimize(model_loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

epochs = 100
for epoch in range(epochs):
  for batch in range(len(inputs_train)):
    input = np.zeros([seq_len, number_of_ops])
    for i, op in enumerate(inputs_train[batch]):
      if not op is None:
        input[i][op] = 1

    output = np.zeros(seq_len)
    output[ outputs_train[batch] ] = 1
      
    feed_dict = { model_inputs: input, model_outputs: output }
    loss_before = session.run([model_loss], feed_dict)
    _, loss = session.run([model_train, model_loss], feed_dict)
    if loss < loss_before:
      diff = "BETTER"
    else:
      diff = "WORSE"
    print("Epoch {0} loss {1} loss_before {2} diff {3} ".format(epoch, loss, loss_before, diff))

# check it

inputs_test, outputs_test = load_files("test")

right = 0
wrong = 0
for batch in range(len(inputs_test)):
  input = np.zeros([seq_len, number_of_ops])
  for i, op in enumerate(inputs_test[batch]):
    if not op is None:
      input[i][op] = 1

  feed_dict = { model_inputs: input }
  op_idx = session.run(model_op_idx, feed_dict)
  if op_idx == outputs_test[batch]:
    right += 1
  else:
    wrong += 1

print("right({0}) wrong({1})".format(right, wrong))

