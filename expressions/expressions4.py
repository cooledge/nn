# described is at http://gregmcclement.blogspot.ca/2017/03/parsing-natural-languages.html

import tensorflow as tf
import numpy as np
import pdb
import sys

if sys.version_info.major == 2:
  def read_string(message):
    return raw_input(message)
else:
  def read_string(message):
    return input(message)

class ParserModel:

  def __init__(self, priorities):
    self.priorities = priorities
    self.init_data()
    self.get_preference_model()
    self.get_parser_model()

  def from_op_to_id(self, op):
    try:
      if op in self.op_to_id.keys():
        return(self.op_to_id[op])
      else:
        return(self.op_to_id['c'])
    except:
      return(self.op_to_id['c'])
     
  def init_data(self):
    ops = set([])
    for tuple in self.priorities:
      ops.add(tuple[0])
      ops.add(tuple[1])

    self.id_to_op = {}
    self.op_to_id = {}
    for op in ops:
      id = len(self.id_to_op)
      self.id_to_op[id] = op
      self.op_to_id[op] = id

    self.number_of_ops = len(ops)
    batch_size = len(self.priorities)
    self.inputs = np.zeros((batch_size, self.number_of_ops))
    self.labels = np.zeros((batch_size, self.number_of_ops))
    batch_no = -1
    for k, v in self.priorities:
      batch_no = batch_no + 1
      self.labels[batch_no][self.from_op_to_id(k)] = 1.0
      self.inputs[batch_no][self.from_op_to_id(v)] = 1.0

  def get_preference_model(self):
    self.model_inputs = tf.placeholder(tf.float32, shape=(None, self.number_of_ops), name="inputs")
    self.model_labels = tf.placeholder(tf.float32, shape=(None, self.number_of_ops), name="labels")
    #model_w = tf.get_variable("w", shape=(self.number_of_ops, self.number_of_ops), dtype=tf.float32)
    model_w = tf.Variable(tf.zeros((self.number_of_ops, self.number_of_ops)))
    #model_b = tf.get_variable("b", shape=(self.number_of_ops), dtype=tf.float32)
    model_b = tf.Variable(tf.zeros((self.number_of_ops)))
    model_logits =  tf.matmul(self.model_inputs, model_w) + model_b
    self.model_loss = tf.losses.softmax_cross_entropy(self.model_labels, model_logits)
    self.model_predict = tf.nn.softmax(model_logits)
    model_optimizer = tf.train.AdamOptimizer(0.05)
    self.model_train_op = model_optimizer.minimize(self.model_loss)

  def get_parser_model(self):
    model_combined = tf.squeeze(tf.reshape(tf.reduce_sum(tf.split(self.model_predict, self.number_of_ops, axis=1), 1), (1, -1)))
    model_reduced = tf.nn.relu(self.model_inputs - model_combined)
    self.model_op_idx = tf.arg_max(tf.reduce_sum(model_reduced, 1), 0)

  def train(self, epochs = 50):
    for epoch in range(epochs):
      feed_dict = { self.model_inputs: self.inputs, self.model_labels: self.labels }
      _, loss = session.run([self.model_train_op, self.model_loss], feed_dict)
      print("Epoch {0} loss {1}".format(epoch, loss))

  def model_op_ids(self):
    return self.model_op_ids

  def get_op(self, expression):
    inputs = np.zeros((len(expression), self.number_of_ops))
    for i in range(len(expression)):
      inputs[i][self.from_op_to_id(expression[i])] = 1
    op_idx = session.run(self.model_op_idx, feed_dict = { self.model_inputs: inputs })

    return op_idx

priorities = [
  ('to', 'a'),
  ('to', 'the'),
  ('move', 'to'),
  ('bought', 'to'),
  ('c', 'move'),
  ('c', 'bought'),
  ('0', 'c')
]

'''
priorities = [
  ('preposition', 'article'),
  ('infix verb', 'preposition'),
  ('constant', 'infix verb'),
  ('0', 'c')
]

words = {
  ('to', 'preposition'),
  ('from', 'preposition'),
  ('move', 'infix verb'),
  ('bought', 'infix verb')
}

'''

parser_model = ParserModel(priorities)

session = tf.Session()
session.run(tf.global_variables_initializer())

parser_model.train()

while True:
  ex_string = read_string("Enter an sentence if you dare: ")
  if ex_string == "":
    break

  expression = ex_string.split()
  print("Input Expression: {0}".format(expression))
  while len(expression) > 1:
    op_idx = parser_model.get_op(expression)
    op = expression[op_idx]

    if op == 'c':
      break
    
    if op == 'bought':
      before = expression[:op_idx-1]
      after = expression[op_idx+2:]
      l = expression[op_idx - 1]
      r = expression[op_idx + 1]
      result = { "action": "buy", "buyer": l, "thing": r }
    elif op == 'move':
      before = expression[:op_idx]
      after = expression[op_idx+3:]
      r = expression[op_idx + 1]
      to = expression[op_idx + 2]
      result = { "action": 'move', "thing": r, "to": to["thing"] }
    elif op == 'to':
      before = expression[:op_idx]
      after = expression[op_idx+2:]
      r = expression[op_idx + 1]
      result = { "preposition": op, "thing": r }
    elif op == 'a' or op == 'the':
      before = expression[:op_idx]
      after = expression[op_idx+2:]
      r = expression[op_idx + 1]
      result = { "determiner": op, "thing": r }
    else:
      break

    expression = before + [result] + after

  for e in expression:
    if isinstance(e, dict):
      print("Output Expression: {0}".format(e))
  



