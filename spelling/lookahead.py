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
#filename = './data/small.txt'
filename = './data/regina.txt'
max_len_word = 20
max_len_phrase = 3

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
class TokenSequence(keras.utils.Sequence):

  def __init__(self, x, y, n_tokens, batch_size):
    self.x = x
    self.y = y
    self.batch_size = batch_size
    self.n_tokens = n_tokens

  def __len__(self):
    return math.ceil(len(self.x) /self.batch_size)

  def __getitem__(self, idx):
    start = idx*self.batch_size
    end = start + self.batch_size
    #batch_x = to_one_hot(self.x[start:end], vocab_size)
    batch_x = to_one_hot(self.x[start:end], self.n_tokens)
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

generator_train_words = TokenSequence(train_x_words, train_y_words, vocab_size, batch_size)
generator_validation_words = TokenSequence(validation_x, validation_y_words, vocab_size, batch_size)
generator_test_words = TokenSequence(test_x_words, test_y_words, vocab_size, batch_size)

words_model_path = "models/words.h5py"

def character_to_word_model():
  model = keras.Sequential()
  # produce output of 128 
  model.add(keras.layers.LSTM(128, input_shape=(max_len_word, vocab_size)))
  model.add(keras.layers.Dense(num_words))
  model.add(keras.layers.Activation('softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def check_all():
  generator = TokenSequence(inputs, outputs, vocab_size, batch_size)
  score, acc = model.evaluate_generator(generator)
  print("Check All Score: {} Accuracy: {}".format(score, acc))

try:
  model = keras.models.load_model(words_model_path)
  score, acc = model.evaluate_generator(generator_test_words)
  print("After load weights Score: {} Accuracy: {}".format(score, acc))
except:
  model = character_to_word_model()
  model.fit_generator(generator_train_words, epochs=50, validation_data=generator_validation_words)
  keras.models.save_model(model, words_model_path)
  score, acc = model.evaluate_generator(generator_test_words)
  print("Score: {} Accuracy: {}".format(score, acc))

check_all()

class PhraseModel: 

  def setup_phrases(max_len):
    phrases = []
    phrase = []
    for line in lines:
      for word in keras.preprocessing.text.text_to_word_sequence(line):
        phrase.append(word_to_id[word])
        if len(phrase) == max_len:
          phrases.append(phrase.copy())
          phrase.pop(0)

    phrases_in = phrases.copy()[:-1]
    phrases_out = phrases
    phrases_out.pop(0)
    return (phrases_in, phrases_out)


  def word_to_phrase_model(max_len):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(128, input_shape=(max_len, num_words)))
    model.add(keras.layers.RepeatVector(max_len))
    model.add(keras.layers.LSTM(128, return_sequences=True))
    #model.add(keras.layers.TimeDistributed(keras.layers.Dense(256, activation=keras.activations.relu)))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_words)))
    model.add(keras.layers.Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  def __init__(self, max_len):
    phrases_in, phrases_out = PhraseModel.setup_phrases(max_len)
    self.model = PhraseModel.word_to_phrase_model(max_len)
     
    phrases_in_one_hot = np.array(to_one_hot(phrases_in, num_words))
    phrases_out_one_hot = np.array(to_one_hot(phrases_out, num_words))
    self.model.fit(phrases_in_one_hot, phrases_out_one_hot, epochs=100, batch_size=8) 

  def predict(self, x):
    return self.model.predict(x, verbose=0)

phrase_models = [PhraseModel(i+1) for i in range(max_len_phrase)]

def predict_add_one(phrase, expected_len):
  max_len = min(len(phrase), max_len_phrase)
  model = phrase_models[max_len-1]
  pred_phrase = phrase[-max_len:]
  x_pred = np.zeros((1, max_len, num_words))
  for t, word in enumerate(pred_phrase):
    if word in word_to_id:
      x_pred[0, t, word_to_id[word]] = 1.
  preds = model.predict(x_pred)[0]
  preds = [np.argmax(one_hot_char) for one_hot_char in preds]
  return phrase + [id_to_word[preds[len(pred_phrase)-1]]]

def predicts_add_one(phrase, prob, n_choices, expected_len):
  max_len = min(len(phrase), max_len_phrase)
  model = phrase_models[max_len-1]
  pred_phrase = phrase[-max_len:]
  x_pred = np.zeros((1, max_len, num_words))
  for t, word in enumerate(pred_phrase):
    if word in word_to_id:
      x_pred[0, t, word_to_id[word]] = 1.
  preds = model.predict(x_pred)[0]

  # get the top n_choices
  max_preds = np.argsort(preds, 1)
  phrase_idx = len(pred_phrase)-1
  next_words = max_preds[phrase_idx][-n_choices:]
  return [(prob*preds[phrase_idx][idx], phrase + [id_to_word[idx]]) for idx in next_words]

# phrases like of (prob, phrase)
def predicts_extend_one(phrases, n_choices, expected_len):
  next_phrases = []
  for (prob, phrase) in phrases:
    next_phrases += predicts_add_one(phrase, prob, n_choices, expected_len)
  return next_phrases

def predicts_extend(phrase, n_choices, expected_len):
  phrases = [(1., phrase)]
  for i in range(expected_len):
    phrases = predicts_extend_one(phrases, n_choices, expected_len)
    phrases = [pair for pair in phrases if pair[0] > 0.01]
  return phrases
   
def predict_extend(phrase, expected_len):
  while len(phrase) < expected_len:
    phrase = predict_add_one(phrase, expected_len)
  return phrase

while True:
  line = input('Enter a phrase: ')
  if line == '':
    break
  phrase = keras.preprocessing.text.text_to_word_sequence(line)
  predict = predict_extend(phrase, 5)
  print("Prediction: {}".format(predict))
  phrases = predicts_extend(phrase, 2, 3)
  phrases = sorted(phrases, key=lambda phrase: phrase[0])
  print(phrases)


