#import tensorflow as tf
from tensorflow import keras
import pickle
#import keras
import math
import random
import pdb
import os

import numpy as np

max_len = 20

lines = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

words = set([])
chars = set([])
for line in lines:
  for word in keras.preprocessing.text.text_to_word_sequence(line):
    for ch in word:
      chars.add(ch)
    words.add(word)

id_to_char = [' '] + list(chars)
char_to_id = { char:id for (id, char) in enumerate(id_to_char) }
vocab_size = len(id_to_char)
words = [list(word)[0:max_len] for word in words]

id_to_word = []
word_to_id = {}
for id,word in enumerate(words):
  word = ''.join(word)
  id_to_word.append(word)
  word_to_id[word] = id
num_words = len(id_to_word)

def word_chars_to_ids(word):
  return [char_to_id[char] for char in word]

def words_to_io(words):
  inputs = []
  outputs = []
  for word in words:
    inputs.append(word)
    word_id = word_to_id[''.join(word)]
    outputs.append(word_id)
  inputs = [word_chars_to_ids(word) for word in inputs]
  inputs = keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len, padding='post', value=char_to_id[" "])
  return (np.array(inputs), np.array(outputs))

inputs, outputs = words_to_io(words)

def get_model():
  model = keras.Sequential()
  # produce output of 128 
  model.add(keras.layers.LSTM(128, input_shape=(max_len, vocab_size)))
  model.add(keras.layers.Dense(num_words))
  model.add(keras.layers.Activation('softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def id_to_one_hot(tid, size):
  one_hot = np.zeros((size,))
  one_hot[tid] = 1.0
  return one_hot

def output_to_one_hot(output, size):
  return [id_to_one_hot(id, size) for id in output]

def to_one_hot(values, size):
  return np.array([output_to_one_hot(value, size) for value in values])

batch_size = 4
class WordSequence(keras.utils.Sequence):

  def __init__(self, x, y, batch_size):
    self.x = x
    self.y = y
    self.batch_size = batch_size

  def __len__(self):
    return math.ceil(len(self.x) /self.batch_size)

  def __getitem__(self, idx):
    start = idx*self.batch_size
    end = start + self.batch_size
    batch_x = to_one_hot(self.x[start:end], vocab_size)
    batch_y = np.array([id_to_one_hot(id, num_words) for id in self.y[start:end]])
    return (np.array(batch_x), np.array(batch_y))

generator_train = WordSequence(inputs, outputs, batch_size)
generator_test = WordSequence(inputs, outputs, batch_size)

checkpoint_path = "words2/model.h5py"

try:
  model = keras.models.load_model(checkpoint_path)
except:
  model = get_model()
  model.fit_generator(generator_train, verbose=0, epochs=50)
  keras.models.save_model(model, checkpoint_path)

score, acc = model.evaluate_generator(generator_test)
print("CTEST Accuracy: {}".format(acc))
