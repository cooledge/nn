import tensorflow as tf
from tensorflow import keras
import math
import pdb

import numpy as np

filename = './data/warandpeace.txt'
#filename = './data/small.txt'
max_len = 20

lines = []
with open(filename) as f:
  lines = [line for line in f.readlines()]

words = set([])
chars = set([])
for line in lines:
  for word in tf.keras.preprocessing.text.text_to_word_sequence(line):
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

def words_to_io(words):
  inputs = []
  outputs = []
  for word in words:
    inputs.append(word)
    word_id = word_to_id[''.join(word)]
    outputs.append(word_id)
    for w in words_remove_letter(word):
      inputs.append(w)
      outputs.append(word_id)
    for w in words_transpose_letter(word):
      inputs.append(w)
      outputs.append(word_id)
  return (inputs, outputs)


print("Making input len(words)={}".format(len(words)))
inputs, outputs = words_to_io(words)
print("Done making input")
def word_chars_to_ids(word):
  return [char_to_id[char] for char in word]

#inputs_words = inputs

inputs = [word_chars_to_ids(word) for word in inputs]
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len, padding='post', value=char_to_id[" "])

# word -> correct word

model = keras.Sequential()
# produce output of 128 
model.add(keras.layers.LSTM(128, input_shape=(max_len, vocab_size)))
model.add(keras.layers.Dense(num_words))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

generator = WordSequence(inputs, outputs, batch_size)
'''
inputs = to_one_hot(inputs, vocab_size)
outputs_words = outputs
outputs = np.array([id_to_one_hot(id, num_words) for id in outputs])
pdb.set_trace()
model.fit(inputs, outputs, epochs=50, batch_size=16)
'''
pdb.set_trace()

model.fit_generator(generator, epochs=50)

predictions = model.predict_classes(inputs)

def from_categorical(batch, id_to_token_map):
  return [id_to_word[word_id] for word_id in batch]

predictions = from_categorical(predictions, id_to_word)
expectations = [id_to_word[word_id] for word_id in outputs_words]

diff = [(expected, predicted) for (expected, predicted) in zip(expectations, predictions) if not expected == predicted]
if len(diff) > 0:
  print("The different is {}".format(diff))
else:
  print("All Okay")
'''
def one_hot_to_id(one_hot):
  pdb.set_trace()
  return np.argmax(one_hot)

def one_hot_output_to_ids(output):
  return [one_hot_to_id(one_hot) for one_hot in output]
predictions = [ one_hot_output_to_ids(output) for output in predictions]
'''

'''
model = model.add(keras.layers.LSTM(32))
model = model.add(keras.layers.LSTM(8))
model = model.add(keras.layers.TimeDistributed(Dense(32))
'''
