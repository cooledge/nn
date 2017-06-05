import numpy as np
import tensorflow as tf
import pdb

class ChainModel:

  # input_model implements
  #   inputs() - placeholder for the inputs
  #   input_to_id(input) - converts the input to an id number
  #   number_of_classes - number of ids 
  #   outputs - the node that calculates the outputs
  #  
  # output_model implements
  #   outputs() - the node that calculates the outputs 
  #   setup(inputs) - create the inference model

  def __init__(self, input_model, output_model):
    self.input_model = input_model
    self.output_model = output_model

  def inputs(self):
    return self.input_model.inputs()

  def outputs(self):
    return self.output_model.outputs()

  # inputs is an array of ids. this corresponds to a sequence 

  def apply(self, session, inputs):
    
    number_of_classes = self.input_model.number_of_classes
    one_hot_inputs = np.zeros((len(inputs), number_of_classes), np.float32)
    for i in range(len(inputs)):
      one_hot_inputs[i][self.input_model.input_to_id(inputs[i])] = 1.0

    feed_dict = { self.inputs(): one_hot_inputs }

    return session.run(self.outputs(), feed_dict)

  def train(self, session):
    self.input_model.train(session)
    self.output_model.train(session)

    self.input_model.setup()
    self.output_model.setup(self.input_model.outputs())
