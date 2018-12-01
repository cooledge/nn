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

if not os.path.exists('models'):
  os.makedirs('models')

lines = []
with open(filename) as f:
  lines = [line for line in f.readlines()]

words = set([])
chars = set([])

def line_to_words(line):
  words = []
  for word in keras.preprocessing.text.text_to_word_sequence(line):
    for ch in word:
      chars.add(ch)
    words.append(word)
  return words

for line in lines:
  words |= set(line_to_words(line))

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
    return math.floor(len(self.x) /self.batch_size)

  def __getitem__(self, idx):
    start = idx*self.batch_size
    end = start + self.batch_size
    batch_x = to_one_hot(self.x[start:end], self.n_tokens)
    batch_y = np.array([id_to_one_hot(id, num_words) for id in self.y[start:end]])
    return (np.array(batch_x), np.array(batch_y))

class TokensSequence(keras.utils.Sequence):

  def __init__(self, x, y, n_tokens, seq_len, batch_size):
    self.x = x
    self.y = y
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.chunk_size = batch_size*seq_len
    self.n_tokens = n_tokens

  def __len__(self):
    return math.floor(len(self.x)/(self.chunk_size))

  def __getitem__(self, idx):
    start = idx*self.chunk_size
    # 16,20,43 -> batch_size, max_len_word, vocab_size
    # wanted -> 16, 3, 20, 43
    batch_x = np.zeros((self.batch_size, self.seq_len, max_len_word, self.n_tokens))
    for batch_no in range(self.batch_size):
      for word_no in range(self.seq_len):
        batch_x[batch_no, word_no] = output_to_one_hot(self.x[start], self.n_tokens)
        start += 1
    
    # 16, 262 -> batch_size, num_words
    # wanted -> 16, 3, 262
    start = idx*self.chunk_size
    batch_y = np.zeros((self.batch_size, self.seq_len, num_words))
    for batch_no in range(self.batch_size):
      for word_no in range(self.seq_len):
        batch_y[batch_no, word_no] = id_to_one_hot(self.y[start], num_words)
        start += 1
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

generator_train_wordss = TokensSequence(train_x_words, train_y_words, vocab_size, max_len_phrase, batch_size)
generator_validation_wordss = TokensSequence(validation_x, validation_y_words, vocab_size, max_len_phrase, batch_size)
generator_test_wordss = TokensSequence(test_x_words, test_y_words, vocab_size, max_len_phrase, batch_size)
#pdb.set_trace()
#len23 = generator_train_wordss.__len__()
#generator_train_wordss.__getitem__(len23-1)

word_model_path = "models/word.h5py"
words_model_path = "models/words.h5py"
phrases_model_path = "models/phrase{0}.h5py"

class WordModel:

  def embedding_size():
    return 64

  def character_to_word_model():
    model = keras.Sequential()
    model.add(keras.layers.LSTM(WordModel.embedding_size(), input_shape=(max_len_word, vocab_size), name='embedding'))
    #model.add(keras.layers.LSTM(128, input_shape=(max_len_word, vocab_size)))
    #model.add(keras.layers.Dense(WordModel.embedding_size(), name='embedding'))
    #model.add(keras.layers.ReLU())
    model.add(keras.layers.Dense(num_words))
    model.add(keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  def __init__(self, train=True):
    if train and os.path.isfile(word_model_path):    
      self.model = keras.models.load_model(word_model_path)
      score, acc = self.model.evaluate_generator(generator_test_words)
      print("After load weights Score: {} Accuracy: {}".format(score, acc))
    else:
      self.model = WordModel.character_to_word_model()
      if (train):
        self.model.fit_generator(generator_train_words, epochs=1, validation_data=generator_validation_words)
        keras.models.save_model(self.model, word_model_path)
        score, acc = self.model.evaluate_generator(generator_test_words)
        print("Score: {} Accuracy: {}".format(score, acc))

  def save(self):
    keras.models.save_model(self.model, word_model_path)

  def check_all(self):
    generator = TokenSequence(inputs, outputs, vocab_size, batch_size)
    score, acc = self.model.evaluate_generator(generator)
    print("Check All Score: {} Accuracy: {}".format(score, acc))

  def line_to_input(self, line):
    words = []
    for word in keras.preprocessing.text.text_to_word_sequence(line):
      words.append(word)

    inputs = [word_chars_to_ids(word) for word in words]
    inputs = keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len_word, padding='post', value=char_to_id[" "])
    inputs = to_one_hot(inputs, vocab_size)
    return inputs

  def softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  def id_to_embedding(self, word_id):
    word = id_to_word[word_id]
    return self.predict_embedding(word)[0]
    #return self.softmax(self.predict_embedding(word)[0])

  def predict_embedding(self, line):
    inputs = self.line_to_input(line)
    model_e = keras.Model(self.model.inputs, self.model.get_layer('embedding').output)
    return model_e.predict([inputs])

  def predict(self, line):
    inputs = self.line_to_input(line)
    preds = self.model.predict([inputs])
    return [id_to_word[np.argmax(pred)] for pred in preds]

  def line_to_embedded(self, line):
    return self.predict_embedding(line)

  def to_embedded(self, lines):
    return np.array([self.line_to_embedded(line) for line in lines])


class WordsModel:

  def __init__(self, max_len):
    if os.path.isfile(word_model_path):    
      self.word_model = WordModel(train=False);
      self.model = keras.models.load_model(words_model_path)
      score, acc = self.model.evaluate_generator(generator_test_wordss)
      print("After load weights Score: {} Accuracy: {}".format(score, acc))
    else:
      self.word_model = WordModel(train=False);
      input_shape = self.word_model.model.input.shape
      inputs_shape = (max_len,) + tuple(input_shape)[1:]
      input_words = keras.layers.Input(shape=inputs_shape)
      output_words = keras.layers.TimeDistributed(self.word_model.model)(input_words)
      self.model = keras.Model(input_words, output_words)
      self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

      self.model.fit_generator(generator_train_wordss, epochs=100, validation_data=generator_validation_wordss)
      self.word_model.save()
      keras.models.save_model(self.model, words_model_path)
      score, acc = self.model.evaluate_generator(generator_test_wordss)
      print("Score: {} Accuracy: {}".format(score, acc))

  def check_all(self):
    generator = TokensSequence(inputs, outputs, vocab_size, max_len_phrase, batch_size)
    score, acc = self.model.evaluate_generator(generator)
    print("Check All Score: {} Accuracy: {}".format(score, acc))
    print("Original word model")
    self.word_model.check_all()

#word_model = WordModel()
#pdb.set_trace()
words_model = WordsModel(max_len_phrase)
pdb.set_trace()
words_model.check_all()
#pdb.set_trace()
#pred = word_model.predict_embedding("reina wascnaa")
#em = word_model.to_embedded(['regina wascana'])
#embedding_size = word_model.embedding_size()

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
    # phrases_in like [ [203, 205], [206, 210] ... ]
    phrases_in_embedded = np.array([[word_model.id_to_embedding(word_id) for word_id in phrase] for phrase in phrases_in])

    phrases_out = phrases
    phrases_out.pop(0)
    phrases_out_one_hot = np.array(to_one_hot(phrases_out, num_words))

    return (phrases_in_embedded, phrases_out_one_hot)

  def word_to_phrase_model(max_len):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(128, input_shape=(max_len, WordModel.embedding_size())))
    model.add(keras.layers.RepeatVector(max_len))
    model.add(keras.layers.LSTM(128, return_sequences=True))
    #model.add(keras.layers.TimeDistributed(keras.layers.Dense(256, activation=keras.activations.relu)))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_words)))
    model.add(keras.layers.Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  def __init__(self, max_len):
    try:
      self.model = keras.models.load_model(phrases_model_path.format(max_len))
    except:
      phrases_in_embedded, phrases_out_one_hot = PhraseModel.setup_phrases(max_len)
      self.model = PhraseModel.word_to_phrase_model(max_len)
      self.model.fit(phrases_in_embedded, phrases_out_one_hot, epochs=100, batch_size=16) 
      keras.models.save_model(self.model, phrases_model_path.format(max_len))

  def predict(self, x):
    return self.model.predict(x, verbose=0)

phrase_models = [PhraseModel(i+1) for i in range(max_len_phrase)]

def predict_add_one(phrase_embedded, phrase, expected_len):
  max_len = min(len(phrase_embedded), max_len_phrase)
  model = phrase_models[max_len-1]
  pred_phrase_embedded = np.expand_dims(phrase_embedded[-max_len:], axis=0)
  preds = model.predict(pred_phrase_embedded)[0]
  preds = [np.argmax(one_hot_char) for one_hot_char in preds]
  preds_embedded = [word_model.predict_embedding(id_to_word[pred])[0] for pred in preds]
  return phrase_embedded + preds_embedded, phrase + [id_to_word[preds[max_len-1]]]

def predicts_add_one(phrase_embedded, phrase, prob, n_choices, expected_len):
  max_len = min(len(phrase), max_len_phrase)
  model = phrase_models[max_len-1]
  pred_phrase = phrase[-max_len:]
  pred_phrase_embedded = np.expand_dims(phrase_embedded[-max_len:], axis=0)
  preds = model.predict(pred_phrase_embedded)[0]

  # get the top n_choices
  max_preds = np.argsort(preds, 1)
  phrase_idx = len(pred_phrase)-1
  next_words = max_preds[phrase_idx][-n_choices:]
  return [
    (
      prob*preds[phrase_idx][idx], 
      [i for i in phrase_embedded] + [i for i in word_model.predict_embedding(id_to_word[idx])],
      phrase + [id_to_word[idx]]
    ) 
    for idx in next_words
  ]

# phrases like of (prob, phrase)
def predicts_extend_one(phrases, n_choices, expected_len):
  next_phrases = []
  for (prob, phrase_embedded, phrase) in phrases:
    next_phrases += predicts_add_one(phrase_embedded, phrase, prob, n_choices, expected_len)
  return next_phrases

def predicts_extend(phrase_embedded, phrase, n_choices, expected_len):
  phrases = [(1., phrase_embedded, phrase)]
  for i in range(expected_len):
    phrases = predicts_extend_one(phrases, n_choices, expected_len)
    phrases = [node for node in phrases if node[0] > 0.01]
  return phrases
   
def predict_extend(phrase_embedded, phrase, expected_len):
  while len(phrase) < expected_len:
    phrase_embedded, phrase = predict_add_one(phrase_embedded, phrase, expected_len)
  return phrase

while True:
  line = input('Enter a phrase: ')
  if line == '':
    break
  phrase_embedded = word_model.predict_embedding(line)
  phrase = word_model.predict(line)
  predict = predict_extend(phrase_embedded, phrase, 5)
  print("Prediction: {}".format(predict))
  phrases = predicts_extend(phrase_embedded, phrase, 2, 3)
  phrases = sorted(phrases, key=lambda phrase: phrase[0])
  for phrase in phrases:
    print(phrase[2])


