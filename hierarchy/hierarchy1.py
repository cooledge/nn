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

types = set(hierarchy.keys()) | set(hierarchy.values())
id_to_name = {}
name_to_id = {}
for type in types:
  id = len(id_to_name)
  id_to_name[id] = type
  name_to_id[type] = id

child_to_parent = {}
for child in hierarchy:
  child_to_parent[name_to_id[child]] = name_to_id[hierarchy[child]]

number_of_classes = len(id_to_name)
number_of_examples = len(child_to_parent)

input = np.zeros((number_of_examples, number_of_classes), np.float32)
output = input.copy()
counter = 0
for c in child_to_parent:
  p = child_to_parent[c]
  input[counter][c] = 1.0 
  output[counter][p] = 1.0 
  counter += 1

def get_model(batch_size):
  model_input = tf.placeholder(tf.float32, shape=[batch_size, number_of_classes], name="input")
  model_output = tf.placeholder(tf.float32, shape=[batch_size, number_of_classes], name="output")

  model_weights = tf.get_variable("weights", shape=[number_of_classes, number_of_classes])
  model_biases = tf.get_variable("biases", shape=[number_of_classes])

  model_logits = tf.matmul(model_input, model_weights) + model_biases
  model_pred = tf.nn.softmax(model_logits)
  model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_output, logits=model_logits))
  model_opt = tf.train.GradientDescentOptimizer(0.5)
  model_train_op = model_opt.minimize(model_loss)

  return([model_train_op, model_input, model_output, model_loss, model_pred])

def get_hmodel():
  model_weights = tf.get_variable("weights", shape=[number_of_classes, number_of_classes])
  model_biases = tf.get_variable("biases", shape=[number_of_classes])
 
  input = tf.placeholder(tf.float32, [1, number_of_classes])
  scaler = 10.0

  def layer(input):
    return(tf.nn.softmax(tf.scalar_mul(scaler, tf.matmul(input, model_weights) + model_biases)))

  l1 = layer(input)
  l2 = layer(l1)
  l3 = layer(l2)
  l4 = layer(l3)
  
  output = tf.clip_by_value(input + l1 + l2 + l3 + l4, 0.0, 1.0)

  return([input, output])

batch_size = number_of_examples

with tf.variable_scope("train") as scope:
  model_train_op, model_input, model_output, model_loss, model_pred = get_model(batch_size)

session = tf.Session()

session.run(tf.global_variables_initializer())

epochs = 200
feed = { model_input: input, model_output: output }
for epoch in range(epochs):
  _, loss = session.run([model_train_op, model_loss], feed_dict=feed)
  if epoch % 10 == 0:
    print("Epoch %d, loss is %d" % (epoch, loss))

pred = session.run(model_pred, feed)
right = 0
for i in range(number_of_examples):
  p = np.argmax(pred[i])
  a = np.argmax(output[i])
  if p == a:
    right += 1
  else:
    print("Wrong pred = %s actual = %s" % (id_to_name[p], id_to_name[a]))

print("Success Rate %d/%d" % (right, number_of_examples))

# hierarchical classifier

with tf.variable_scope("train") as scope:
  scope.reuse_variables()
  classifier_input, classifier_output = get_hmodel()

  print("Valid type are: ") 
  for type in name_to_id.keys():
    print("\t%s" % (type))
  print("")

  while True:
    type = raw_input("Enter type type: ")

    if not type in name_to_id.keys():
      break

    input = np.zeros((1, number_of_classes), np.float32)
    id = name_to_id[type]
    input[0][id] = 1

    feed = { classifier_input: input }
    output = session.run( classifier_output, feed )

    def display(title, output):
      print(title)
      for i in range(len(output)):
        if output[i] > 0.50: 
          print("%s has prob %f" % (id_to_name[i], output[i]))
      print("")

    display("Output (prob > 50%)", output[0])

