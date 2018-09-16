#import tensorflow as tf
from tensorflow import keras
import pickle
#import keras
import math
import random
import pdb
import os

import numpy as np

#filename = './data/warandpeace.txt'
filename = './data/small.txt'
max_len_word = 20

lines = []
with open(filename) as f:
  lines = [line for line in f.readlines()]

words = set([])
chars = set([])

for line in lines:
  for word in keras.preprocessing.text.text_to_word_sequence(line):
    for ch in word:
      chars.add(ch)
    words.add(word)

# make sure the id's are the same between runs

words = sorted(words)
chars = sorted(chars) 

id_to_char = [' '] + list(chars)
char_to_id = { char:id for (id, char) in enumerate(id_to_char) }
vocab_size = len(id_to_char)
words = [list(word)[0:max_len_word] for word in words]

id_to_word = []
word_to_id = {}
for id,word in enumerate(words):
  word = ''.join(word)
  id_to_word.append(word)
  word_to_id[word] = id
num_words = len(id_to_word)

def word_remove_letter(word, i):
  word = word[:]
  del(word[i])
  return word

assert list("abd") == word_remove_letter(list("abcd"), 2)

# skip first letter
def words_remove_letter(word):
  return [word_remove_letter(word, i+1) for i in range(len(word)-1)]

assert [list('acd'), list('abd'), list('abc')] == words_remove_letter(list("abcd"))

def word_transpose_letter(word, i):
  word = word[:]
  ch = word[i]
  word[i] =  word[i+1]
  word[i+1] = ch
  return word

assert list("acbd") == word_transpose_letter(list("abcd"), 1)

def words_transpose_letter(word):
  return [word_transpose_letter(word, i) for i in range(len(word)-1)]

assert [list("bacd"), list("acbd"), list("abdc")] == words_transpose_letter(list("abcd"))

def word_chars_to_ids(word):
  return [char_to_id[char] for char in word]

def words_to_io(words, include_misspellings=True):
  inputs = []
  outputs = []
  for word in words:
    inputs.append(word)
    word_id = word_to_id[''.join(word)]
    outputs.append(word_id)
    if include_misspellings:
      for w in words_remove_letter(word):
        inputs.append(w)
        outputs.append(word_id)
      for w in words_transpose_letter(word):
        inputs.append(w)
        outputs.append(word_id)
  inputs = [word_chars_to_ids(word) for word in inputs]
  inputs = keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len_word, padding='post', value=char_to_id[" "])
  return (np.array(inputs), np.array(outputs))

print("Making input len(words)={}".format(len(words)))
inputs, outputs = words_to_io(words)

indexes = [i for i in range(len(inputs))]
random.shuffle(indexes)
inputs = inputs[indexes]
outputs = outputs[indexes]

print("Done making input")

#inputs_words = inputs

# word -> correct word

def id_to_one_hot(tid, size):
  one_hot = np.zeros((size,))
  one_hot[tid] = 1.0
  return one_hot

def output_to_one_hot(output, size):
  return [id_to_one_hot(id, size) for id in output]

def to_one_hot(values, size):
  return np.array([output_to_one_hot(value, size) for value in values])

batch_size = 16
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

def split_percents(values, percentages):
  n = len(values)
  numbers = [int(0.01*percentage*n) for percentage in percentages]
  splits = []
  start = 0
         
  for number in numbers:
    splits.append(values[start:start+number])
    start += number
  return splits
  
train_x_words, validation_x, test_x_words = split_percents(inputs, [80, 10, 10])                      
train_y_words, validation_y_words, test_y_words = split_percents(outputs, [80, 10, 10])                  

def ids_to_string(chars):
  return ''.join([id_to_char[id] for id in chars]).strip()

def x_to_words(xs):
  return [ids_to_string(x) for x in xs]

def y_to_words(ys):
  return [id_to_word[y] for y in ys]

generator_train_words = WordSequence(train_x_words, train_y_words, batch_size)
generator_validation_words = WordSequence(validation_x, validation_y_words, batch_size)
generator_test_words = WordSequence(test_x_words, test_y_words, batch_size)

checkpoint_path = "words2/model.h5py"

def character_to_word_model():
  model = keras.Sequential()
  # produce output of 128 
  model.add(keras.layers.LSTM(128, input_shape=(max_len_word, vocab_size)))
  model.add(keras.layers.Dense(num_words))
  model.add(keras.layers.Activation('softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def word_to_phrase_model():
  model = keras.Sequential()
  model.add(keras.layers.LSTM(128, input_shape=(max_len_phrase, number_of_words)))
  model.add(keras.layers.RepeatVector(max_len_phrase))
  model.add(keras.layers.LSTM(128, return_sequences=True))
  model.add(keras.layers.TimeDistributed(keras.layers.Dense(number_of_words)))
  model.add(keras.layers.Activation("softmax"))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  
try:
  model = keras.models.load_model(checkpoint_path)
  score, acc = model.evaluate_generator(generator_test_words)
  print("After load weights Score: {} Accuracy: {}".format(score, acc))
except:
  model = character_to_word_model()
  model.fit_generator(generator_train_words, epochs=50, validation_data=generator_validation_words)
  keras.models.save_model(model, checkpoint_path)
  score, acc = model.evaluate_generator(generator_test_words)
  print("Score: {} Accuracy: {}".format(score, acc))

