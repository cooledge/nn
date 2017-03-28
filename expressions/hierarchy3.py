import tensorflow as tf
import numpy as np
import re
import sys
import math
import os
import pdb
from six.moves import cPickle as pickle

class HierarchyModel:

  def __init__(self, hierarchy, default, hierarchy_outputs):
    self.init_data(hierarchy, hierarchy_outputs)
    self.default = default
    self.init_model()
    #self.init_hmodel()

  # Interface for chain

  def inputs(self):
    return self.model_input

  def outputs(self):
    return self.model_hierarchy_output

  def input_to_id(self, input):
    try:
      if input in self.name_to_id:
        return self.name_to_id[input]
      else:
        return self.name_to_id[self.default]
    except:
      return self.name_to_id[self.default]

  def number_of_classes(self):
    return self.number_of_classes

  def setup(self):
    self.init_hmodel()

  # End Interface for chain

  def init_data(self, hierarchy, hierarchy_outputs):
    types = set([])
    for tuple in hierarchy:
      types.add(tuple[0])
      types.add(tuple[1])

    self.id_to_name = {}
    self.name_to_id = {}

    # setup the outputs so they are first in the one hot vector
    for type in hierarchy_outputs:
      id = len(self.id_to_name)
      self.id_to_name[id] = type
      self.name_to_id[type] = id
    self.number_of_outputs = len(hierarchy_outputs)

    for type in types:
      if type in self.name_to_id:
        continue
      id = len(self.id_to_name)
      self.id_to_name[id] = type
      self.name_to_id[type] = id

    self.child_to_parent = {}
    for tuple in hierarchy:
      child = tuple[0] 
      parent = tuple[1] 
      self.child_to_parent[self.name_to_id[child]] = self.name_to_id[parent]

    self.number_of_classes = len(self.id_to_name)

  def init_model(self):
    self.model_input = tf.placeholder(tf.float32, shape=[None, self.number_of_classes], name="hierarchy_input")
    self.model_labels = tf.placeholder(tf.float32, shape=[None, self.number_of_classes], name="hierarchy_labels")

    self.model_weights = tf.Variable(tf.zeros([self.number_of_classes, self.number_of_classes]))
    self.model_biases = tf.Variable(tf.zeros([self.number_of_classes]))

    model_logits = tf.matmul(self.model_input, self.model_weights) + self.model_biases
    self.model_pred = tf.nn.softmax(model_logits)
    self.model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.model_labels, logits=model_logits))
    model_opt = tf.train.GradientDescentOptimizer(0.5)
    self.model_train_op = model_opt.minimize(self.model_loss)

  def init_hmodel(self):
    scaler = 10.0

    def layer(input):
      return(tf.nn.softmax(tf.scalar_mul(scaler, tf.matmul(input, self.model_weights) + self.model_biases)))

    l1 = layer(self.model_input)
    l2 = layer(l1)
    l3 = layer(l2)
    l4 = layer(l3)
   
    self.model_hierarchy_output_full = tf.clip_by_value(self.model_input + l1 + l2 + l3 + l4, 0.0, 1.0)
    self.model_hierarchy_output = tf.split(tf.clip_by_value(self.model_input + l1 + l2 + l3 + l4, 0.0, 1.0), [self.number_of_outputs, (self.number_of_classes-self.number_of_outputs)], axis=1)[0]

  #def model_hierarchy_output():
    #self.model_hierachy_output

  def infer(self, session, type):
    input = np.zeros((1, self.number_of_classes), np.float32)
    id = self.name_to_id[type]
    input[0][id] = 1

    feed = { self.model_input: input }
    return session.run( self.model_hierarchy_output, feed )

  def train(self, session):

    number_of_examples = len(self.child_to_parent)

    input = np.zeros((number_of_examples, self.number_of_classes), np.float32)
    labels = input.copy()
    counter = 0
    for c in self.child_to_parent:
      p = self.child_to_parent[c]
      input[counter][c] = 1.0 
      labels[counter][p] = 1.0 
      counter += 1

    epochs = 200
    feed = { self.model_input: input, self.model_labels: labels }
    for epoch in range(epochs):
      _, loss = session.run([self.model_train_op, self.model_loss], feed_dict=feed)
      if epoch % 10 == 0:
        print("Epoch %d, loss is %d" % (epoch, loss))

    pred = session.run(self.model_pred, feed)
    right = 0
    for i in range(number_of_examples):
      p = np.argmax(pred[i])
      a = np.argmax(labels[i])
      if p == a:
        right += 1
      else:
        print("Wrong pred = %s actual = %s" % (self.id_to_name[p], self.id_to_name[a]))

    print("Success Rate %d/%d" % (right, number_of_examples))

  def classes(self):
    return self.name_to_id.keys()

if __name__ == "__main__":
  raw_input = input

  hierarchy = [ 
    ("bear", "mammal"),
    ("tiger", "mammal"),
    ("whale", "mammal"),
    ("ostrich", "bird"),
    ("peacock", "bird"),
    ("eagle", "bird"),
    ("salmon", "fish"),
    ("goldfish", "fish"),
    ("guppy" , "fish"),
    ("turtle", "reptile"),
    ("crocodile", "reptile"),
    ("snake", "reptile"),
    ("frog", "amphibian"),
    ("toad", "amphibian"),
    ("ant", "invertibrates"),
    ("cockroach", "invertibrates"),
    ("ladybird", "invertibrates"),
    ("mammal", "vertibrates"),
    ("birds", "vertibrates"),
    ("fish", "vertibrates"),
    ("reptiles", "vertibrates"),
    ("amphibians", "vertibrates"),
    ("vertibrates", "animals"),
    ("invertibrates", "animals"),
    ("animals", "animals")  # loop for things with no higher class
  ]

  houtputs = ['mammal', 'bird', 'fish', 'reptile', 'amphibian', 'invertibrates', 'vertibrates', 'animals']

  hierarchy_model = HierarchyModel(hierarchy, "animals", houtputs)
  hierarchy_model.setup()

  session = tf.Session()
  session.run(tf.global_variables_initializer())

  hierarchy_model.train(session)

  # hierarchical classifier

  print("Valid type are: ") 
  for type in hierarchy_model.classes():
    print("\t%s" % (type))
  print("")

  while True:
    type = raw_input("Enter type type: ")

    if not type in hierarchy_model.classes():
      break

    output = hierarchy_model.infer(session, type)

    def display(title, output):
      print(title)
      for i in range(len(output)):
        if output[i] > 0.50: 
          print("%s has prob %f" % (hierarchy_model.id_to_name[i], output[i]))
      print("")

    display("Output (prob > 50%)", output[0])

