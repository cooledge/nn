# work in progress. I want to work on dealing with ambiguity and see what happens

import tensorflow as tf
import numpy as np
import sys
import pdb
from hierarchy4 import HierarchyModel
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

def apply_move_chess_piece(op, op_idx, expression, cds):
  before = expression[:op_idx]
  after = expression[op_idx+3:]
  r = expression[op_idx + 1]
  to = expression[op_idx + 2]
  result = { "action": 'move_chess_piece', "piece": r, "square": to["thing"] }
  return before + [result] + after

def apply_bought(op, op_idx, expression, cds):
  before = expression[:op_idx-1]
  after = expression[op_idx+2:]
  l = expression[op_idx - 1]
  r = expression[op_idx + 1]
  result = { "action": "buy", "buyer": l, "thing": r }
  return before + [result] + after

def apply_preposition(op, op_idx, expression, cds):

  '''  
  expression2 = cds.current()
  before2 = [ tuple[1] for tuple in expression2[:op_idx]]
  after2 = [ tuple[1] for tuple in expression2[op_idx+2:]]
  r2 = expression2[op_idx + 1][1]
  result2 = { "preposition": op, "thing": r2 }
  result2 = before2 + [result2] + after2
  '''
 
  before = expression[:op_idx]
  after = expression[op_idx+2:]
  r = expression[op_idx + 1]
  result = { "preposition": op, "thing": r }
  result1 = before + [result] + after
  pdb.set_trace()
  return result1

def increment_depth(tuple):
  return (tuple[0]+1, tuple[1])

def apply_article(op, op_idx, expression, cds):
  '''
  current = cds.current()
  pdb.set_trace()
  cds.joins( [current[op_idx][0], current[op_idx+1][0]], increment_depth(current[op_idx][0]), "result to be set" )
  '''
  '''
  expression = cds.current()
  before = [tuple[1] for tuple in expression[:op_idx]]
  after = [tuple[1] for tuple in expression[op_idx+2:]]
  r = expression[op_idx + 1][1]
  result = { "determiner": op, "thing": r }
  return before + [result] + after
  '''
  before = expression[:op_idx]
  after = expression[op_idx+2:]
  r = expression[op_idx + 1]
  result = { "determiner": op, "thing": r }
  return before + [result] + after

op_to_apply = { 
  "move": apply_move_chess_piece, 
  "bought": apply_bought, 
  "to": apply_preposition, 
  "a": apply_article, 
  "the": apply_article 
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

class CDS:

  def __init__(self, session, length, depth):
    self.session = session
    self.length = length
    self.depth = depth
    self.values = [ [None for _ in range(length)] for _ in range(depth+1) ]
    self.placeholders = [ tf.placeholder(tf.float32, shape=(1, length)) for _ in range(depth) ]

    self.nodes = []
    self.layers = []
    
    self.nodes.append(tf.constant([[1]], dtype=tf.float32))

    m = tf.Variable(tf.zeros([1,length]), name="m1")
    self.layers.append(m)
    self.nodes.append(tf.matmul(self.nodes[-1], self.layers[-1]))

    il = 0
    for d in range(depth):
      il = il + length
      m = tf.Variable(tf.zeros([il, length]), name="m{0}".format(d+2))
      self.layers.append(m)
      i = tf.reshape([self.nodes[1:]], (1, il))
      self.nodes.append(tf.matmul(i, self.layers[-1]))

    self.leaves = []
  
    for d in range(len(self.nodes)-1): 
      self.debug = []
      s = tf.clip_by_value(self.nodes[d+1], 0, 1)
      self.debug.append(s)
      for m in self.layers[d+1:]:
        l = tf.slice(m, [d*length,0], [length,length])
        r = tf.reduce_sum(l, [1]) * -1
        s = s + r
        self.debug.append(s)
      self.leaves.append(s)
   
  def show(self):
    nodes, leaves, debug, layers = self.session.run([self.nodes, self.leaves, self.debug, self.layers])

    print("Nodes")
    for row in nodes:
      for column in row:
        print(column)

    print("Leaves")
    for row in leaves:
      for column in row:
        print(column)

    print("Values")
    for row in range(len(self.values)):
      for col in range(len(self.values[row])):
        if self.values[row][col]:
          print("[{0}][{1}]: {2}".format(row, col, self.values[row][col]))

    print("Current")
    print(self.current())
    '''
    print("Debug")
    for l in debug:
      print(l)

    print("Layers")
    for i in range(len(layers)):
      print("{0}: {1}".format(i, layers[i]))
    '''

    print("") 

  def in_graph(session):
    #session.run(self.nodes[1:], { self.nodes[0]: 1 })
    session.run(self.nodes[1:])
 
  #def is_leaf():

  # join nodes at depth into the next level to node 
  def join(self, from_row, from_col, to_row, to_col): 
    layer = self.layers[to_row]
    update = np.zeros(layer.get_shape())
    update[from_row*self.length + from_col][to_col] = 1.0
    self.session.run(layer.assign_add(update))

  def joins(self, froms, to, value):
    for f in froms:
      self.join(f[0], f[1], to[0], to[1])
    self.values[to[0]][to[1]] = value

  def current(self):
    current = []
    leaves = self.session.run(self.leaves)
    for row in range(len(leaves)):
      for col in range(len(leaves[0][0])):
        if leaves[row][0][col] > 0:
          current.append(((row,col), self.values[row][col]))
    return current

''' 
session2 = tf.Session()
length = 4
depth = 3
cds = CDS(session2, length, depth)
session2.run(tf.global_variables_initializer())
#cds.show()
cds.join(0, 0, 0, 0, "value10")
cds.join(0, 0, 0, 1, "value10")
cds.join(0, 0, 0, 2, "value10")

cds.join(0, 0, 2, 1, "value10")
cds.join(0, 1, 2, 1, "value10")

cds.join(0, 2, 3, 1, "value10")
cds.join(2, 1, 3, 1, "value10")
cds.show()
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

  for i in range(len(expression)):
    cds.joins( [(0,0)], (0,i), expression[i] )
  cds.show() 

  print("Input Expression: {0}".format(expression))
  while len(expression) > 1:
    op_idx = chain_model.apply(session, expression)
    op = expression[op_idx]

    if op == 'constant':
      break
 
    expression = op_to_apply[op](op, op_idx, expression, cds) 
  return expression

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
  
