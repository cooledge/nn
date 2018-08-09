# implement char-rnn in tensorflow

import os
import argparse
import collections
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
#from conv1d_inverse import conv1d_inverse
from tensorflow.contrib import legacy_seq2seq as seq2seq
#from tensorflow.python.ops import seq2seq
import pdb

# implement char_rnn_tf 

# this works: python ae_text.py --epochs 100 --stop_at 0.001 --type words --batch_len 20 --beam_len 10 --no-load
parser = argparse.ArgumentParser(description="Program to train a story generator")
parser.add_argument("--data", default="./data", type=str, help="directory with input.txt")
parser.add_argument("--seq_len", default=10, type=int, help="length of the sequences for the rnn")
parser.add_argument("--epochs", default=50, type=int, help="nummer of epichs tu run")
parser.add_argument("--stop_at", default=0.0, type=float, help="stops training when loss is less than this value")
parser.add_argument("--batch_len", default=20, type=int, help="length of the rnn")
parser.add_argument("--predict_len", default=5, type=int, help="length of the generated output")
parser.add_argument("--beam_len", default=1, type=int, help="size of beam for prediction")
parser.add_argument("--no-load", default=False, help="load the saved model", action='store_true')
parser.add_argument("--repeat", default=25, type=int, help="how many times to repeat the input data")

args = parser.parse_args()
print(args.data)

batch_len = args.batch_len
batch_size = 32

epochs = args.epochs

model_dir = os.path.dirname(os.path.abspath(__file__)) + "/model_char"
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
model_filename = model_dir + "/model"

batch_size_in_chars = batch_size*batch_len
punctuation = ',!.;:'

# get the input and output data
def load_file(file_dir, token_type):
  file = open(os.path.join(file_dir, "input.txt"), "r")
  data = file.read()
  data = data.lower()
  data = data*25
  data = data.replace('*', '')
  if token_type == 'chars':
    # okay
    0
  elif token_type == 'words':
    for p in punctuation:
      data = data.replace(p, ' {0} '.format(p))
    data = data.replace('\n', ' <newline> ')
    data = data.split()

  # make the input and target

  counter = collections.Counter(data)
  number_of_tokens = len(counter.keys())
  token_to_id_map = {}
  id_to_token_map = {}
  for i, token in enumerate(counter.keys()):
    token_to_id_map[token] = i
    id_to_token_map[i] = token
  inputs = []

  for token in data:
    inputs.append(token_to_id_map[token])
  inputs = np.array(inputs)
  targets = inputs.copy()

  number_of_batches = len(inputs)//batch_size_in_chars

  def setup_inputs(inputs):
    inputs = inputs[0:(number_of_batches*batch_size_in_chars)]
    inputs = np.reshape(inputs, [batch_size, -1])
    inputs = np.split(inputs, number_of_batches, 1)
    return inputs

  # (60, 50)
  inputs = setup_inputs(inputs)
  # (60, 50)
  targets = setup_inputs(targets)
  '''
  i_s = []
  t_s = []
  for batch_i,batch_t in zip(inputs, targets):
    for element_i,element_t in zip(batch_i, batch_t):
    i_s.append(i)
    t_s.append(t)
    pdb.set_trace()
    for j in range(len(i_s)):
      miss = i.copy()
      miss.pop(j)
      miss.append(token_to_id_map[' '])
      i_s.append(miss)
      t_s.append(t)
  '''

  return([inputs, targets, number_of_tokens, token_to_id_map, id_to_token_map])

input, targets, number_of_tokens, token_to_id_map, id_to_token_map = load_file(args.data, 'chars')

# setup the model
embedding_size = 128

def model(batch_size, batch_len, number_of_tokens, reuse): 
  model = keras.Sequential()
  model.add(keras.layers.Embedding(number_of_tokens, embedding_size))
  model.add(keras.layers.Conv1D(64, 2, padding='same'))
  model.add(keras.layers.MaxPool1D(2, 1, padding='same'))
  model.add(keras.layers.Dense(embedding_size))

  model_batch_size = tf.Variable(batch_size, trainable=False)
  W = tf.get_variable("W", shape=(embedding_size, number_of_tokens))
  b = tf.get_variable("b", shape=(number_of_tokens))

  target_placeholder = tf.placeholder(tf.int32, shape=(None, batch_len), name="target")
  outputs = model.output
  outputs = tf.reshape(outputs, [-1, embedding_size])
  input_placeholder = model.input
  # (3000, number_of_tokens) 
  logits = tf.matmul(outputs, W) + b
  #probs = tf.nn.softmax(logits, 1, name="probs")
  probs = tf.nn.softmax(logits, -1, name="probs")

  loss = seq2seq.sequence_loss_by_example([logits], [tf.reshape(target_placeholder, [-1])], [tf.ones([batch_size * batch_len])], number_of_tokens)
  return([loss, probs, input_placeholder, target_placeholder, logits, model_batch_size])

def optimizer(batch_size, batch_len, loss):
  model_lr = tf.Variable(0.0, trainable=False)
  tvars = tf.trainable_variables()

  optimizer = tf.train.AdamOptimizer(model_lr)
  cost_op = tf.reduce_sum(loss) / batch_size / batch_len
  '''
  grads= tf.gradients(cost_op, tvars)
  grad_clip = 5
  tf.clip_by_global_norm(grads, grad_clip)
  grads_and_vars = zip(grads, tvars)
  train_op = optimizer.apply_gradients(grads_and_vars)
  '''
  train_op = optimizer.minimize(loss)

  return [train_op, cost_op, model_lr]

loss, probs, input_placeholder, target_placeholder, logits, model_batch_size = model(batch_size, batch_len, number_of_tokens, False)
train_op, cost_op, model_lr = optimizer(batch_size, batch_len, loss)

# train the model

session = tf.Session()
session.run(tf.global_variables_initializer())

saver = tf.train.Saver()

if not args.no_load:
  try:
    saver.restore(session, model_filename)
  except Exception as e:
    0 # ignore 

learning_rate = 0.002
#decay = 0.97
decay = 1.0
for epoch in range(epochs):
  sys.stdout.write("Epoch {0}".format(epoch+1))
  session.run(tf.assign(model_lr, learning_rate * (decay ** epoch)))
  for i, t in zip(input, targets):
    feed = {input_placeholder: i, target_placeholder: t}
    cost, _, lr = session.run([cost_op, train_op, model_lr], feed_dict=feed)
  print(" {0} lr({1})".format(cost, lr))
  if cost < args.stop_at:
    break
saver.save(session, model_filename)

# sample the model
pdb.set_trace()
#loss, probs, input_placeholder, target_placeholder, last_state, model_batch_size = model(1, batch_len, number_of_tokens, True)

def predict(predictions, n_samples, beam_size):
# now the super hacky part. Generate one character at a time. If we generate
# exactly what the RNN says the output is very repetitive so instead of doing
# that we get the probabilities for each output and randomly select a character
# based on its likelyhood. That makes the output not boring. 
# FUTURE WORK: get the hacky bit in the neural network

  def hacky_character_picker( probs, beam_size ):
    #beam_size = 3
    if beam_size == 1:
      cs = np.cumsum(probs)
      t = np.sum(probs)
      r = np.random.rand(1)
      cutoff = r * t
      return [(int(np.searchsorted(cs, cutoff)), 1.0)]
    else:
      tokens = [(i, probs[0][i]) for i in range(len(probs[0]))]
      tokens = sorted(tokens, key=lambda data: -data[1])
      tokens = tokens[:beam_size]
      return tokens
  
  def get_next(result, state, result_prob, actual_probs):
    tokens = hacky_character_picker(actual_probs, beam_size)
    nexts = []
    for id, token_prob in tokens:
      token = id_to_token_map[id]
      next_result = result.copy()
      next_result.append(token)
      i = [[id]]
      
      feed = {input_placeholder: i}

      next_actual_probs, next_state = session.run([probs, last_state], feed_dict=feed)
      nexts.append((next_result, next_state, result_prob*token_prob, next_actual_probs))
    return nexts
  
  result, state, prob, actual_probs = predictions[0]

  for i in range(n_samples):
    next_p = []
    for p in predictions:
      next_p += get_next(*p)
    predictions = sorted(next_p, key=lambda p: -p[2])
    predictions = predictions[:beam_size]

  return [''.join(result) for result,_,_,_ in predictions]

# result, state, token
def setup_predictions(prime):
  result = [ id_to_token_map[i] for i in prime]
  state = session.run(decoder_initial_state)
  for id in prime:
    i = [[id]]
    t = [[id]]
     
    feed = {input_placeholder: i, decoder_initial_state: state}

    actual_probs, state = session.run([probs, last_state], feed_dict=feed)
  prob = 1.0
  return [(result, state, prob, actual_probs)]

def autoencoder_predict(prime):
  input = []
  for i in range(batch_len):
    if i < len(prime):
      input.append(prime[i])
    else:
      input.append(0)
  feed = {input_placeholder: [input], model_batch_size: 1}

  actual_probs = session.run(probs, feed_dict=feed)
  return [id_to_token_map[np.argmax(cprob)] for cprob in actual_probs]

print("Enter some starter text")
for line in sys.stdin:
  if line == '\n':
    break
  line = line.replace('\n', '')
  input = [ token_to_id_map[t] for t in line]
  n_samples = 100
  print('-'*80)
  prediction = autoencoder_predict(input)
  print(prediction)
  print("Enter some starter text")

