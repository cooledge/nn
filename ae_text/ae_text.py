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

  inputs = inputs[0:(number_of_batches*batch_size_in_chars)]
  inputs = [inputs[i:i + batch_len] for i in range(0, len(inputs), batch_len)]
  targets = inputs

  return([inputs, targets, number_of_tokens, token_to_id_map, id_to_token_map])

input, targets, number_of_tokens, token_to_id_map, id_to_token_map = load_file(args.data, 'chars')

# setup the model
embedding_size = 128

def model(batch_size, batch_len, number_of_tokens, reuse): 
  model = keras.Sequential()
  model.add(keras.layers.Embedding(number_of_tokens, embedding_size))
  model.add(keras.layers.Conv1D(64, 2, padding='same'))
  model.add(keras.layers.MaxPool1D(2, 1, padding='same'))
  model.add(keras.layers.Dense(embedding_size, name='to_emdedding', activation='relu'))
  model.add(keras.layers.Dense(number_of_tokens, name='to_tokens', activation='softmax'))
  #model.add(keras.layers.Reshape((number_of_tokens,)))
  model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')  
  return model

model = model(batch_size, batch_len, number_of_tokens, False)

pdb.set_trace()
ctargets = tf.keras.utils.to_categorical(targets, num_classes=number_of_tokens)
#ctargets = np.reshape(ctargets, (-1, batch_len*number_of_tokens))
model.fit([targets], [ctargets], epochs=args.epochs, batch_size=batch_size)

print("Enter some starter text")
for line in sys.stdin:
  if line == '\n':
    break
  line = line.replace('\n', '')
  input = [ token_to_id_map[t] for t in line]
  n_samples = 100
  print('-'*80)
  pdb.set_trace()
  prediction = model.predict([[input]], batch_size=1)
  prediction = np.argmax(prediction, axis=1)
  print(prediction)
  print("Enter some starter text")

