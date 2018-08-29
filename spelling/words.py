import tensorflow as tf
from tensorflow import keras

import numpy as np

words = [] # list of words made up of ids

def word_remove_letter(word, i):
  return del(word[:][i])

# skip first letter
def words_remove_letter(word):
  return [word_remove_letter(word, i) for i+1 in range(len(word)-1)]

def word_transpose_letter(word, i)
  word = word[:]
  ch = word[i]
  word[i] =  word[i+1]
  word[i+1] = ch
  return word

def words_transpose_letter(word):
  return [word_transpose_letter(word, i) for i in range(len(word)-1)]

# word -> correct word
