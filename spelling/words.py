import tensorflow as tf
from tensorflow import keras
import pdb

import numpy as np

filename = './data/warandpeace.txt'
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

inputs = [word_chars_to_ids(word) for word in inputs]
outputs = [word_chars_to_ids(word) for word in outputs]

inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len, padding='post', value=char_to_id[" "])
outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs, maxlen=max_len, padding='post', value=char_to_id[" "])

pdb.set_trace()
# word -> correct word
'''
model = keras.Sequential()
model = model.add(keras.layers.Embedding(vocab_size, 16))
model = model.add(keras.layers.
'''
