import tensorflow as tf
import numpy as np
import sys
import pdb
from chain1 import ChainModel

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

