import tensorflow as tf
from tensorflow import keras
import pdb

import numpy as np

words = [list("abcd")] # list of words made up of ids

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
print(inputs)
print(outputs)

# word -> correct word
