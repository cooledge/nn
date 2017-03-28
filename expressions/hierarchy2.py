import tensorflow as tf
import numpy as np
import re
import sys
import math
import os
import pdb
from six.moves import cPickle as pickle

raw_input = input

hierarchy = { 
  "bear": "mammal",
  "tiger": "mammal",
  "whale": "mammal",
  "ostrich": "bird",
  "peacock": "bird",
  "eagle": "bird",
  "salmon": "fish",
  "goldfish": "fish",
  "guppy" : "fish",
  "turtle": "reptile",
  "crocodile": "reptile",
  "snake": "reptile",
  "frog": "amphibian",
  "toad": "amphibian",
  "newt": "amphibian",
  "ant": "invertibrates",
  "cockroach": "invertibrates",
  "ladybird": "invertibrates",
  "mammal": "vertibrates",
  "birds": "vertibrates",
  "fish": "vertibrates",
  "reptiles": "vertibrates",
  "amphibians": "vertibrates",
  "vertibrates": "animals",
  "invertibrates": "animals",
  "animals": "animals"  # loop for things with no higher class
}

class HierarchyModel:

  def __init__(self, hierarchy):
    self.init_data()
    self.init_model()
    self.init_hmodel()


  def init_data(self):
    types = set(hierarchy.keys()) | set(hierarchy.values())
    self.id_to_name = {}
    self.name_to_id = {}
    for type in types:
      id = len(self.id_to_name)
      self.id_to_name[id] = type
      self.name_to_id[type] = id

    self.child_to_parent = {}
    for child in hierarchy:
      self.child_to_parent[self.name_to_id[child]] = self.name_to_id[hierarchy[child]]

    self.number_of_classes = len(self.id_to_name)


  def init_model(self):
    self.model_input = tf.placeholder(tf.float32, shape=[None, self.number_of_classes], name="input")
    self.model_output = tf.placeholder(tf.float32, shape=[None, self.number_of_classes], name="output")

    self.model_weights = tf.Variable(tf.zeros([self.number_of_classes, self.number_of_classes]))
    self.model_biases = tf.Variable(tf.zeros([self.number_of_classes]))

    model_logits = tf.matmul(self.model_input, self.model_weights) + self.model_biases
    self.model_pred = tf.nn.softmax(model_logits)
    self.model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.model_output, logits=model_logits))
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
    
    self.model_hierarchy_output = tf.clip_by_value(self.model_input + l1 + l2 + l3 + l4, 0.0, 1.0)

  def infer(self, type):
    input = np.zeros((1, self.number_of_classes), np.float32)
    id = self.name_to_id[type]
    input[0][id] = 1

    feed = { self.model_input: input }
    return session.run( self.model_hierarchy_output, feed )

  def train(self):
    number_of_examples = len(self.child_to_parent)

    input = np.zeros((number_of_examples, self.number_of_classes), np.float32)
    output = input.copy()
    counter = 0
    for c in self.child_to_parent:
      p = self.child_to_parent[c]
      input[counter][c] = 1.0 
      output[counter][p] = 1.0 
      counter += 1

    epochs = 200
    feed = { self.model_input: input, self.model_output: output }
    for epoch in range(epochs):
      _, loss = session.run([self.model_train_op, self.model_loss], feed_dict=feed)
      if epoch % 10 == 0:
        print("Epoch %d, loss is %d" % (epoch, loss))

    pred = session.run(self.model_pred, feed)
    right = 0
    for i in range(number_of_examples):
      p = np.argmax(pred[i])
      a = np.argmax(output[i])
      if p == a:
        right += 1
      else:
        print("Wrong pred = %s actual = %s" % (self.id_to_name[p], self.id_to_name[a]))

    print("Success Rate %d/%d" % (right, number_of_examples))

  def classes(self):
    return self.name_to_id.keys()

hierarchy_model = HierarchyModel(hierarchy)

session = tf.Session()
session.run(tf.global_variables_initializer())

hierarchy_model.train()

# hierarchical classifier

print("Valid type are: ") 
for type in hierarchy_model.classes():
  print("\t%s" % (type))
print("")

while True:
  type = raw_input("Enter type type: ")

  if not type in hierarchy_model.classes():
    break

  output = hierarchy_model.infer(type)

  def display(title, output):
    print(title)
    for i in range(len(output)):
      if output[i] > 0.50: 
        print("%s has prob %f" % (hierarchy_model.id_to_name[i], output[i]))
    print("")

  display("Output (prob > 50%)", output[0])

