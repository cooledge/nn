# this version has the current state of the parse represented in a NN
# using the CDS data structure. 

import tensorflow as tf
import numpy as np
import sys
import pdb
from hierarchy4 import HierarchyModel
from cds2 import CDS
from chain1 import ChainModel

class ParserModel:

  def __init__(self, priorities, default_priority, one_hot_inputs):
    self.priorities = priorities
    self.default_priority = default_priority
    self.one_hot_inputs = one_hot_inputs
    self.init_data()
    self.get_preference_model()
    #self.get_parser_model()

  def setup(self, inputs):
    self.get_parser_model(inputs)

  def outputs(self):
    return self.model_op_idx

  def from_op_to_id(self, op):
    try:
      if op in self.op_to_id.keys():
        return(self.op_to_id[op])
      else:
        return(self.op_to_id[default_priority])
    except:
      return(self.op_to_id[default_priority])
     
  def init_data(self):
    self.id_to_op = {}
    self.op_to_id = {}

    for op in self.one_hot_inputs:
      id = len(self.id_to_op)
      self.id_to_op[id] = op
      self.op_to_id[op] = id

    self.number_of_ops = len(self.one_hot_inputs)
    batch_size = len(self.priorities)
    self.inputs = np.zeros((batch_size, self.number_of_ops))
    self.labels = np.zeros((batch_size, self.number_of_ops))
    batch_no = -1
    for k, v in self.priorities:
      batch_no = batch_no + 1
      self.labels[batch_no][self.from_op_to_id(k)] = 1.0
      self.inputs[batch_no][self.from_op_to_id(v)] = 1.0

  def get_preference_model(self):
    self.model_inputs = tf.placeholder(tf.float32, shape=(None, self.number_of_ops), name="expression_inputs")
    self.model_labels = tf.placeholder(tf.float32, shape=(None, self.number_of_ops), name="expression_labels")
    self.model_w = tf.Variable(tf.zeros((self.number_of_ops, self.number_of_ops)))
    self.model_b = tf.Variable(tf.zeros((self.number_of_ops)))
    model_logits =  tf.matmul(self.model_inputs, self.model_w) + self.model_b
    self.model_loss = tf.losses.softmax_cross_entropy(self.model_labels, model_logits)
    self.model_predict = tf.nn.softmax(model_logits)
    model_optimizer = tf.train.AdamOptimizer(0.05)
    self.model_train_op = model_optimizer.minimize(self.model_loss)

  def get_parser_model(self, model_inputs = None):
    if model_inputs == None:
      model_inputs = self.model_inputs
    model_logits =  tf.matmul(model_inputs, self.model_w) + self.model_b
    # predict what is lower priority
    self.model_predict = tf.nn.softmax(model_logits)
    self.model_combined = tf.squeeze(tf.reshape(tf.reduce_sum(tf.split(self.model_predict, self.number_of_ops, axis=1), 1), (1, -1)))
    self.model_reduced = tf.nn.relu(model_inputs - self.model_combined)
    self.model_op_idx = tf.arg_max(tf.reduce_sum(self.model_reduced, 1), 0)

  def train(self, session, epochs = 50):
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

    return session.run(self.model_op_idx, feed_dict = { self.model_inputs: inputs })

priorities = [
  ('preposition', 'article'),
  ('infix', 'preposition'),
  ('constant', 'infix'),
  ('0', 'constant')
]

words = [
  ('constant', 'constant'),
  ('to', 'preposition'),
  ('from', 'preposition'),
  ('move', 'infix'),
  ('bought', 'infix'),
  ('a', 'article'),
  ('the', 'article'),

  # top off the loop
  ('article', 'article'),
  ('preposition', 'preposition'),
  ('infix', 'infix')
]

one_hot_spec = ['constant', 'preposition', 'infix', 'article', '0']
hierarchy_model = HierarchyModel(words, 'constant', one_hot_spec)

parser_model = ParserModel(priorities, 'constant', one_hot_spec)

def get_next_level(elements):
  return max([element[0] for element in elements]) + 1
 
def apply_move_chess_piece(op_idx, expression, cds):
  op = expression[op_idx]

  result = do_op(op_idx, [], "action", ["piece", "square"], cds, 
              {
                op_idx: lambda e: "move_chess_piece",
                op_idx+2: lambda to: to["thing"]
              })

  before = expression[:op_idx]
  after = expression[op_idx+3:]
  return before + [result] + after

def apply_bought(op_idx, expression, cds):
  op = expression[op_idx]

  result = do_op(op_idx, ["buyer"], "action", ["thing"], cds, {op_idx: lambda e: "buy"})

  before = expression[:op_idx-1]
  after = expression[op_idx+2:]
  return before + [result] + after

def do_op(op_idx, before_tags, op_tag, after_tags, cds, overrides = {}):
  expression2 = cds.current()

  tag_to_index = { op_tag: op_idx }
  for idx, after_tag in enumerate(after_tags):
    tag_to_index[after_tag] = op_idx + idx + 1
  for idx, before_tag in enumerate(before_tags):
    tag_to_index[before_tag] = op_idx - idx - 1

  result = {}
  for tag in tag_to_index:
    idx = tag_to_index[tag]
    if idx in overrides:
      result[tag] = overrides[idx](expression2[idx][1])
    else:
      result[tag] = expression2[idx][1] 

  op_pos = expression2[op_idx][0]
  joins = [ expression2[idx][0] for idx in tag_to_index.values() ]
  next_level = get_next_level(joins)
  cds.joins(joins, (next_level, op_pos[1]), result)
  return result

def apply_preposition(op_idx, expression, cds):
  op = expression[op_idx]
  result = do_op(op_idx, [], "preposition", ["thing"], cds)
 
  before = expression[:op_idx]
  after = expression[op_idx+2:]
  r = expression[op_idx + 1]
  result1 = before + [result] + after
  return result1

def apply_done(op_idx, expression, cds):
  raise "done"

def increment_depth(tuple):
  return (tuple[0]+1, tuple[1])

def apply_article(op_idx, expression, cds):
  op = expression[op_idx]
  result = do_op(op_idx, [], "determiner", ["thing"], cds)
  
  before = expression[:op_idx]
  after = expression[op_idx+2:]
  r = expression[op_idx + 1]
  return before + [result] + after

op_to_apply = { 
  "move": apply_move_chess_piece, 
  "bought": apply_bought, 
  "to": apply_preposition, 
  "a": apply_article, 
  "the": apply_article,
  "constant": apply_done
}

'''
can you run neural nets in reverse?
m1 implies what for arg1 and arg2 arg1 and implies what for m
output of a neural net is one node for each type -> then convert that to a one hot vector ala hierarchy

m1 m2 m3   +    arg1.1 arg1.2   + arg2.1 arg2.2 arg2.3
  move a car by driving
  move a chess piece with a hand
  move from one house to another

move1 after to -> together or not
present + types

nueral net for numbers
proper names

the processing loops sets weights in a fully connection multilayer net. That is used as input to the loop
that is you cool data structure (CDS)

expression -> CDS -> expression -f-> CDS -> expression -f-> CDS

'''

if sys.version_info.major == 2:
  def read_string(message):
    return raw_input(message)
else:
  def read_string(message):
    return input(message)

session = tf.Session()
session.run(tf.global_variables_initializer())

chain_model = ChainModel(hierarchy_model, parser_model)
chain_model.train(session)

def evaluate(string):
  max_length = 10
  max_depth = 6
  session_cds = tf.Session()
  cds = CDS(session_cds, max_length, max_depth)
  session_cds.run(tf.global_variables_initializer())
  expression = string.split()

  cds.initialize(expression)
  cds.show() 

  print("Input Expression: {0}".format(expression))
  while len(expression) > 1:
    op_idx = chain_model.apply(session, expression)
    op = expression[op_idx]

    try: 
      op_to_apply[op](op_idx, expression, cds) 
      expression = [t[1] for t in cds.current()]
    except:
      break

  return [t[1] for t in cds.current()]

expression1 = evaluate("move the boat to the island")
expected1 = [{'action': 'move_chess_piece', 'piece': {'thing': 'boat', 'determiner': 'the'}, 'square': {'thing': 'island', 'determiner': 'the'}}]
assert expression1 == expected1

expression2 = evaluate("move t1 to t2")
expected2 = [{'action': 'move_chess_piece', 'piece': 't1', 'square': 't2'}]
assert expression2 == expected2

while True:
  ex_string = read_string("Enter an sentence if you dare: ")
  if ex_string == "":
    break

  expression = evaluate(ex_string)

  for e in expression:
    if isinstance(e, dict):
      print("Output Expression: {0}".format(e))
  
