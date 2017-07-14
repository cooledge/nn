# described is at http://gregmcclement.blogspot.ca/2017/03/parsing-natural-languages.html

import tensorflow as tf
import numpy as np
import pdb
import random

# LOADING THE OPERATORS FILE

seq_len = 8
operators = "+-*/^"
number_of_ops = len(operators)

operators = "+-*/^"

# hmmm, this did one run for train 1,2,3 test 4 where the results were perfect
'''
More things to try:

  DONE check if the right op is selected just the wrong position
  DONE batch the training

  DONE multi hot -> check this
  DONE multiple by number before softmax
  DONE train the pairwise way then check the loss

  current results: trains really good or really bad. the loss function does not decrease very well
  analyze with the training fails maybe there is a different algorithm
  increase the content size
'''


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
    inputs = file.read().splitlines()

  with open("output_{0}.txt".format(suffix)) as file:
    outputs = file.read().splitlines()

  outputs_all = []
  for i_input, v_input in enumerate(inputs):
    outputs_all.append([i for i, x in enumerate(v_input) if x == outputs[i_input]])

  # map outputs to the index of the operator to process
  outputs = [ inputs[i].index(char) for i, char in enumerate(outputs) ]
 
  inputs = [ chars_to_ids(chars) for chars in inputs ]
  #outputs = [ [char_to_id(char)] for char in outputs ]

  return [inputs, outputs, outputs_all]

inputs_train, outputs_train, outputs_all_train = load_files("train")

# END LOADING THE OPERATORS FILE

batch_size = 1
model_inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, number_of_ops), name="inputs")
# the select op
model_outputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len), name="outputs")

model_inputs_by_batch = [tf.squeeze(model_input) for model_input in tf.split(model_inputs, batch_size, axis=0)]

model_w = tf.get_variable("w", shape=(number_of_ops, number_of_ops), dtype=tf.float32)
model_b = tf.get_variable("b", shape=(number_of_ops), dtype=tf.float32)

model_logits_by_batch =  [tf.matmul(model_input, model_w) + model_b for model_input in model_inputs_by_batch]
model_predict_by_batch = [tf.nn.softmax(model_logits) for model_logits in model_logits_by_batch]

model_combined_by_batch = [tf.squeeze(tf.reshape(tf.reduce_sum(tf.split(model_predict, number_of_ops, axis=1), 1), (1, -1))) for model_predict in model_predict_by_batch]
model_reduced_by_batch = [tf.nn.relu(model_input_by_batch - model_combined) for model_input_by_batch, model_combined in zip(model_inputs_by_batch, model_combined_by_batch)]
model_op_idx_by_batch = [tf.arg_max(tf.reduce_sum(model_reduced, 1), 0) for model_reduced in model_reduced_by_batch]
model_reduced_sum_by_batch = [tf.reduce_sum(model_reduced, 1) for model_reduced in model_reduced_by_batch]

model_reduced = tf.concat(model_reduced_sum_by_batch, axis=0)
model_loss = tf.losses.softmax_cross_entropy(tf.reshape(model_outputs, [batch_size*seq_len]), model_reduced)
#model_loss = tf.losses.softmax_cross_entropy(tf.concat(model_outputs, axis=0), tf.reduce_sum(model_reduced, 1))

model_optimizer = tf.train.AdamOptimizer(0.25)  # this was pretty good
#model_optimizer = tf.train.AdamOptimizer(0.05)
#model_optimizer = tf.train.GradientDescentOptimizer(0.10)
model_train = model_optimizer.minimize(model_loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

epochs = 500
number_of_batches = len(inputs_train) / batch_size

def shuffle_pair(l1, l2):
  l = list(zip(l1, l2))
  random.shuffle(l)
  l1, l2 = zip(*l)
  return [l1, l2]

for epoch in range(epochs):
  inputs_train, outputs_train = shuffle_pair(inputs_train, outputs_train)
  for start_batch in range(number_of_batches):
    inputs = np.zeros([batch_size, seq_len, number_of_ops])
    outputs = np.zeros([batch_size, seq_len])

    for batch in range(batch_size):
      for i, op in enumerate(inputs_train[start_batch+batch]):
        if not op is None:
          inputs[batch][i][op] = 1

      setAllIdx = False
      if setAllIdx:
        for idx in outputs_all_train[start_batch+batch]:
          outputs[batch][idx] = 1
      else:
        outputs[batch][outputs_train[start_batch+batch]] = 1
     
    feed_dict = { model_inputs: inputs, model_outputs: outputs }
    _, loss = session.run([model_train, model_loss], feed_dict)
    print("Epoch {0} loss {1}".format(epoch, loss))

# check it

inputs_test, outputs_test, outputs_all_test = load_files("test")

right = 0
right_op_but_diff_position = 0
wrong = 0
number_of_batches = len(inputs_test) / batch_size
for start_batch in range(number_of_batches):
  print("start_batch: {0}".format(start_batch))
  inputs = np.zeros([batch_size, seq_len, number_of_ops])
  for batch in range(batch_size):
    for i, op in enumerate(inputs_test[start_batch+batch]):
      if not op is None:
        inputs[batch][i][op] = 1

  feed_dict = { model_inputs: inputs }
  op_idx_by_batch = session.run(model_op_idx_by_batch, feed_dict)
  for batch in range(batch_size):
    op_idx = op_idx_by_batch[batch]
    if op_idx == outputs_test[start_batch+batch]:
      right += 1
    else:
      if op_idx in outputs_all_test[start_batch+batch]:
        right_op_but_diff_position += 1
      else:
        wrong += 1

print("right({0}) right_op_but_diff_position({1}) wrong({2})".format(right, right_op_but_diff_position, wrong))

