# implement char-rnn in tensorflow

import os
import argparse
import collections
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import sys
from tensorflow.contrib import legacy_seq2seq as seq2seq
#from tensorflow.python.ops import seq2seq
import pdb

# implement char_rnn_tf 

# this works: python ae_text.py --epochs 100 --stop_at 0.001 --type words --batch_len 20 --beam_len 10 --no-load
parser = argparse.ArgumentParser(description="Program to train a story generator")
parser.add_argument("--data", default="./data", type=str, help="directory with input.txt")
parser.add_argument("--seq_len", default=50, type=int, help="length of the sequences for the rnn")
parser.add_argument("--epochs", default=50, type=int, help="nummer of epichs tu run")
parser.add_argument("--stop_at", default=0.0, type=float, help="stops training when loss is less than this value")
parser.add_argument("--type", default='chars', type=str, help="chars or words")
parser.add_argument("--predict_len", default=5, type=int, help="length of the generated output")
parser.add_argument("--beam_len", default=1, type=int, help="size of beam for prediction")
parser.add_argument("--no-load", default=False, help="load the saved model", action='store_true')
parser.add_argument("--auto_encoder", default=False, help="run as an autoencoder as opposed to a predicter", action='store_true')
parser.add_argument("--repeat", default=25, type=int, help="how many times to repeat the input data")

args = parser.parse_args()
print(args.data)

seq_len = args.seq_len
batch_len = args.seq_len
batch_size = 16
cell_state_size = 128 # in cell
rnn_cells_depth = 1 # cells in each time step

epochs = args.epochs

model_dir = os.path.dirname(os.path.abspath(__file__)) + "/model_{0}".format(args.type)
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
  if args.auto_encoder:
    targets = inputs
  else:
    targets[:-1] = inputs[1:]
    targets[-1] = inputs[0]

  number_of_batches = len(inputs)//batch_size_in_chars

  def setup_inputs(inputs):
    inputs = inputs[0:(number_of_batches*batch_size_in_chars)]
    inputs = np.split(inputs, len(inputs)/seq_len)
    return inputs

  # (60, 50)
  inputs = setup_inputs(inputs)
  # (60, 50)
  targets = setup_inputs(targets)
  return([inputs, targets, number_of_tokens, token_to_id_map, id_to_token_map])

inputs, targets, number_of_tokens, token_to_id_map, id_to_token_map = load_file(args.data, args.type)

# setup the model

def modelk_old(cell_state_size, rnn_cells_depth, batch_size, batch_len, number_of_tokens, reuse): 
  model = keras.Sequential()
  embedding_size = 128
  model.add(keras.layers.Embedding(number_of_tokens, embedding_size, input_length=seq_len))
  model.add(keras.layers.LSTM(cell_state_size, return_sequences=True))
  model.add(keras.layers.RepeatVector(batch_len))
  model.add(keras.layers.TimeDistributed(keras.layers.Dense(number_of_tokens)))
  model.add(keras.layers.Activation("softmax"))
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def modelk(cell_state_size, rnn_cells_depth, batch_size, batch_len, number_of_tokens, reuse): 
  model = keras.Sequential()
  embedding_size = 128
  model.add(keras.layers.Embedding(number_of_tokens, embedding_size, input_length=seq_len))
  model.add(keras.layers.LSTM(cell_state_size, input_shape=(batch_len, number_of_tokens), return_sequences=True))
  model.add(keras.layers.LSTM(cell_state_size, return_sequences=True))
  model.add(keras.layers.TimeDistributed(keras.layers.Dense(number_of_tokens)))
  model.add(keras.layers.Activation("softmax"))
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = modelk(cell_state_size, rnn_cells_depth, batch_size, batch_len, number_of_tokens, False)
ctargets = np.array(tf.keras.utils.to_categorical(targets, num_classes=number_of_tokens))
model.fit([inputs], [ctargets], epochs=args.epochs)

seed_text = 'reg'

ap = np.argsort([1,2,1,5,5])

def predict_to_string(predict): 
  tokens = [np.argmax(p) for p in predict]
  return ''.join([id_to_token_map[token] for token in tokens])

to_do = [(1.0, 'regina ')]
n_choices = 2
min_percent = 0.30
max_len = 20
beam_size = 5
while len(to_do[0][1]) < max_len:
  to_do.sort()
  to_do = to_do[-beam_size:]
  prob, seed = to_do.pop()

  input = [[[token_to_id_map[token] for token in seed] + [token_to_id_map[' ']]*(batch_len-len(seed))]]
  inputs = [id_to_token_map[id] for id in input[0][0]]
  predict = model.predict(input)
  pstring = predict_to_string(predict[0])
  probs = predict[0][len(seed)]
  next_chars = np.argsort(probs)
  next_chars = next_chars[-n_choices:]
  for next_char in next_chars:
    if True or probs[next_char] > min_percent:
      next_prob = probs[next_char]
      next_char = id_to_token_map[next_char]
      to_do.append((prob*next_prob, seed + next_char))

print(to_do)
pdb.set_trace()
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
      
      feed = {input_placeholder: i, decoder_initial_state: state}

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

  #result, state, prob, actual_probs = predictions[0]

  if args.type == 'chars':
    return [''.join(result) for result,_,_,_ in predictions]
  else:
    def to_result(prediction):
      result = prediction[0]
      result = ' '.join(result)
      result = result.replace('<newline>', '\n')
      for p in punctuation:
        result = result.replace(' {0}'.format(p), '.')
      return result
    return [to_result(p) for p in predictions]

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
  state = session.run(decoder_initial_state)
  predict = []
  for id in prime:
    i = [[id]]
    t = [[id]]
     
    feed = {input_placeholder: i, decoder_initial_state: state}

    actual_probs, state = session.run([probs, last_state], feed_dict=feed)
    char = np.argmax(actual_probs)
    predict.append(id_to_token_map[char])
  return predict
'''
if not args.auto_encoder:
  input = [ random.randint(0, number_of_tokens-1) for _ in range(3) ] 
  predictions = setup_predictions(input)
  print("AND the result is: ")
  if args.type == 'chars':
    print(predict(predictions, 500, 1))
  else:
    print(predict(predictions, 100, 1))

print("Enter some starter text")
for line in sys.stdin:
  if line == '\n':
    break
  line = line.replace('\n', '')
  if args.type == 'chars':
    input = [ token_to_id_map[t] for t in line]
    n_samples = 100
  elif args.type == 'words':
    words = line.split()
    input = []
    for word in words:
      if word in token_to_id_map:
        input.append(token_to_id_map[word])
    n_samples = 10
  print('-'*80)
  if args.auto_encoder:
    prediction = autoencoder_predict(input)
    print(prediction)
  else:
    predictions = setup_predictions(input)
    predictions = predict(predictions, n_samples, args.beam_len)

    for i, prediction in enumerate(predictions):
      print("{0}. {1}".format(i,prediction))
  print("Enter some starter text")
'''
