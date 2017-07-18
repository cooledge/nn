import tensorflow as tf
import numpy as np
import pdb
import sys
from hierarchy import HierarchyModel
from chain_model import ChainModel

# modelling birds can fly and penguins can't

if sys.version_info.major == 2:
  def read_string(message):
    return raw_input(message)
else:
  def read_string(message): 
    return input(message)

hierarchy = [ 
  ("bear", "mammal"),
  ("tiger", "mammal"),
  ("whale", "mammal"),
  ("bat", "mammal"),
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
  ("ant", "invertibrate"),
  ("cockroach", "invertibrate"),
  ("ladybird", "invertibrate"),
  ("mammal", "vertibrate"),
  ("bird", "vertibrate"),
  ("penguin", "bird"),
  ("fish", "vertibrate"),
  ("reptiles", "vertibrate"),
  ("amphibians", "vertibrate"),
  ("vertibrate", "animal"),
  ("invertibrate", "animal"),
  ("animal", "animal")  # loop for things with no higher class
]

houtputs = ['bird', 'penguin', 'ostrich', 'bat', 'mammal', 'fish', 'reptile', 'amphibian', 'invertibrate', 'vertibrate', 'animal']

hierarchy_model = HierarchyModel(hierarchy, "animal", houtputs)
hierarchy_model.setup()

# train a NN that maps from classes to fly or not fly node

# not fly / fly
number_of_outputs = 2
model_inputs = tf.placeholder(tf.float32, shape=[None, hierarchy_model.number_of_outputs], name="inputs")
model_outputs = tf.placeholder(tf.float32, shape=[None, number_of_outputs], name="outputs")
model_W = tf.get_variable("fly_W", shape=(hierarchy_model.number_of_outputs, number_of_outputs))
model_b = tf.get_variable("fly_b", shape=(number_of_outputs))
model_logits = tf.matmul(model_inputs, model_W) + model_b
model_predict = tf.nn.softmax(model_logits)
model_loss = tf.losses.softmax_cross_entropy(model_outputs, model_logits)
model_optimizer = tf.train.AdamOptimizer(0.10)
model_train = model_optimizer.minimize(model_loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

hierarchy_model.train(session)

class Classifier:

  # hierarchy_model.number_of_outputs == number_of_inputs
  def __init__(number_of_inputs):
    self.number_of_outputs = 2
    self.model_inputs = tf.placeholder(tf.float32, shape=[None, number_of_inputs], name="inputs")
    self.model_outputs = tf.placeholder(tf.float32, shape=[None, number_of_outputs], name="outputs")
    self.model_W = tf.get_variable("fly_W", shape=(number_of_inputs, number_of_outputs))
    self.model_b = tf.get_variable("fly_b", shape=(number_of_outputs))
    self.model_logits = tf.matmul(self.model_inputs, self.model_W) + self.model_b
    self.model_predict = tf.nn.softmax(self.model_logits)
    self.model_loss = tf.losses.softmax_cross_entropy(self.model_outputs, model_logits)
    self.model_optimizer = tf.train.AdamOptimizer(0.10)
    self.model_train = self.model_optimizer.minimize(self.model_loss)

  def outputs(self):
    return self.model_outputs

  def train(self, session):
    bird_output = hierarchy_model.infer(session, "bird")
    penguin_output = hierarchy_model.infer(session, "penguin")
    ostrich_output = hierarchy_model.infer(session, "ostrich")
    mammal_output = hierarchy_model.infer(session, "mammal")
    bat_output = hierarchy_model.infer(session, "bat")
    vertibrate_output = hierarchy_model.infer(session, "vertibrate")
    animal_output = hierarchy_model.infer(session, "animal")

    inputs = [ bird_output, bat_output, ostrich_output, penguin_output, mammal_output, vertibrate_output, animal_output ]
    inputs = [ input[0] for input in inputs ]
    outputs = [ [1,0] ] * len(inputs)
    outputs[0] = [0,1]
    outputs[1] = [0,1]
    epochs = 200
    for epoch in range(epochs):
      # batch size one
      if False:
        for input, output in zip(inputs, outputs):
          _, loss = session.run([self.model_train, self.model_loss], { self.model_inputs: input, self.model_outputs: output})
          print("loss({0})".format(loss))
      else:
        _, loss = session.run([self.model_train, self.model_loss], { self.model_inputs: inputs, self.model_outputs: outputs})
        print("loss({0})".format(loss))

  def setup(self, input_nodes):
    number_of_outputs = 2
    model_inputs = tf.placeholder(tf.float32, shape=[None, hierarchy_model.number_of_outputs], name="inputs")
    model_outputs = tf.placeholder(tf.float32, shape=[None, number_of_outputs], name="outputs")
    model_W = tf.get_variable("fly_W", shape=(hierarchy_model.number_of_outputs, number_of_outputs))
    model_b = tf.get_variable("fly_b", shape=(number_of_outputs))
    model_logits = tf.matmul(model_inputs, model_W) + model_b
    model_predict = tf.nn.softmax(model_logits)
    model_loss = tf.losses.softmax_cross_entropy(model_outputs, model_logits)
    model_optimizer = tf.train.AdamOptimizer(0.10)
    model_train = model_optimizer.minimize(model_loss)

bird_output = hierarchy_model.infer(session, "bird")
penguin_output = hierarchy_model.infer(session, "penguin")
ostrich_output = hierarchy_model.infer(session, "ostrich")
mammal_output = hierarchy_model.infer(session, "mammal")
bat_output = hierarchy_model.infer(session, "bat")
vertibrate_output = hierarchy_model.infer(session, "vertibrate")
animal_output = hierarchy_model.infer(session, "animal")

inputs = [ bird_output, bat_output, ostrich_output, penguin_output, mammal_output, vertibrate_output, animal_output ]
inputs = [ input[0] for input in inputs ]
outputs = [ [1,0] ] * len(inputs)
outputs[0] = [0,1]
outputs[1] = [0,1]
epochs = 200
for epoch in range(epochs):
  # batch size one
  if False:
    for input, output in zip(inputs, outputs):
      _, loss = session.run([model_train, model_loss], { model_inputs: input, model_outputs: output})
      print("loss({0})".format(loss))
  else:
    _, loss = session.run([model_train, model_loss], { model_inputs: inputs, model_outputs: outputs})
    print("loss({0})".format(loss))

# hierarchical classifier

print("bird: {0}".format(bird_output))
print("penguin: {0}".format(penguin_output))
print("ostrich: {0}".format(ostrich_output))
print("bat: {0}".format(bat_output))
print("vertibrate: {0}".format(vertibrate_output))
print("animal: {0}".format(animal_output))

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
    for i, output in enumerate(output):
      if output > 0.50: 
        print("{0} has prob {1}".format(hierarchy_model.id_to_name[i], output))
    print("")

  predict = session.run(model_predict, { model_inputs: output })
  print("predict: {0}".format(predict))
  if np.argmax(predict) == 1:
    print("FLY")
  else:
    print("NOT FLY")

  display("Output (prob > 50%)", output[0])

