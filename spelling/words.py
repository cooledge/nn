import tensorflow as tf
from tensorflow import keras
import pdb

import numpy as np

#filename = './data/warandpeace.txt'
filename = './data/small.txt'
max_len = 1

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
    outputs.append(word)
    for w in words_remove_letter(word):
      inputs.append(w)
      outputs.append(word)
    for w in words_transpose_letter(word):
      inputs.append(w)
      outputs.append(word)
  return (inputs, outputs)

inputs, outputs = words_to_io(words)

def word_chars_to_ids(word):
  return [char_to_id[char] for char in word]

inputs_words = inputs
inputs = [word_chars_to_ids(word) for word in inputs]
outputs_words = outputs
outputs = [word_chars_to_ids(word) for word in outputs]

inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len, padding='post', value=char_to_id[" "])
outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs, maxlen=max_len, padding='post', value=char_to_id[" "])

# word -> correct word

model = keras.Sequential()
# produce output of 128 
model.add(keras.layers.LSTM(128, input_shape=(max_len, vocab_size)))
# this is the output length not the input length
model.add(keras.layers.RepeatVector(max_len))

# By setting return_sequences to True, return not only the last output but
# all the outputs so far in the form of (num_samples, timesteps,
# output_dim). This is necessary as TimeDistributed in the below expects
# the first dimension to be the timesteps.
model.add(keras.layers.LSTM(128, return_sequences=True))

model.add(keras.layers.TimeDistributed(keras.layers.Dense(vocab_size)))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def id_to_one_hot(tid):
  one_hot = np.zeros((vocab_size,))
  one_hot[tid] = 1.0
  return one_hot

def output_to_one_hot(output):
  return [id_to_one_hot(id) for id in output]

def to_one_hot(values):
  return np.array([output_to_one_hot(value) for value in values])

outputs = to_one_hot(outputs)
inputs = to_one_hot(inputs)
model.fit(inputs, outputs, epochs=50, batch_size=64)

predictions = model.predict_classes(inputs)

def from_categorical(batch, id_to_token_map):
  def sequence_from_categorical(sequence):
    return [id_to_token_map[id] for id in sequence]
  return [sequence_from_categorical(sequence) for sequence in batch]

predictions = [''.join(chars).strip() for chars in from_categorical(predictions, id_to_char)]
expectations = [''.join(chars) for chars in outputs_words]

diff = [(expected, predicted) for (expected, predicted) in zip(expectations, predictions) if not expected == predicted]
'''
def one_hot_to_id(one_hot):
  pdb.set_trace()
  return np.argmax(one_hot)

def one_hot_output_to_ids(output):
  return [one_hot_to_id(one_hot) for one_hot in output]
predictions = [ one_hot_output_to_ids(output) for output in predictions]
'''

pdb.set_trace()

'''
model = model.add(keras.layers.LSTM(32))
model = model.add(keras.layers.LSTM(8))
model = model.add(keras.layers.TimeDistributed(Dense(32))
'''
