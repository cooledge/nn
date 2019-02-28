import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import Counter

'''
load

  get top N words for each language
  generate training data phrases on length L
   
'''

import pdb
from os import listdir
from os.path import isfile, join
import sys
sys.path.append("../lib")
from helpers import splits_by_percentages


def files_in_dir(path):
  return [join(path, f) for f in listdir(path)]

vocab_size = 20000
max_len = 10
percent_training = 0.8
percent_test = 0.1
percent_validification = 0.1
path = 'data'

languages = [f for f in listdir(path)]
id_to_language = languages
language_to_id =  { language:index for (index, language) in enumerate(id_to_language) }
n_languages = len(id_to_language)

language_dirs = files_in_dir(path)

def get_lines(filename, words):
  with open(filename) as f :
    lines = f.readlines()
    for line in lines:
      words.append([word for word in line.lower().split() if word.isalpha()])

lang_to_samples = {}
sample_language = []
sample_text = []
for language in languages:
  lines = []
  for f in listdir(join(path, language)):
    get_lines(join(path, join(language, f)), lines)
  for line in lines:
    if not line == []:
      sample_language.append(language_to_id[language])
      sample_text.append(' '.join(line))

tokenizer = keras.preprocessing.text.Tokenizer(vocab_size)
tokenizer.fit_on_texts(sample_text)

def to_sequences(sample_text):
  sequences = tokenizer.texts_to_sequences(sample_text)
  sequences = keras.preprocessing.sequence.pad_sequences(sequences, padding='post', maxlen=max_len)
  return sequences

sample_text = to_sequences(sample_text)
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 32))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(n_languages, activation=tf.nn.sigmoid))

sample_text_splits, sample_language_splits = splits_by_percentages([sample_text, sample_language], [80,10,10])
sample_text_training, sample_text_validation, sample_text_test = sample_text_splits
sample_language_training, sample_language_validation, sample_language_test = sample_language_splits

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(sample_text_training, sample_language_training, epochs=20, batch_size=100, verbose=1, validation_data=(sample_text_validation, sample_language_validation))

counter = 0
for text in sample_text:
  if sum(text) == 0:
    counter += 1
print(counter)

predictions = model.predict(sample_text)
for prediction, text, language in zip(predictions, sample_text, sample_language):
  if np.argmax(prediction) != language:
    print(tokenizer.sequences_to_texts([text]))


score = model.evaluate(sample_text_test, sample_language_test)
print('Test Loss: ', score[0])
print('Test accuracy: ', score[1])

print("Enter some text")
for line in sys.stdin:
  if line == '\n':
    break

  line = line.replace('\n', '')
  line_of_ngrams = line_to_ngrams(line)
  inputs = to_sequences([line_of_ngrams])
  print('-'*80)
  prediction = model.predict(inputs, batch_size=1)
  prediction = np.argmax(prediction[0])
  print('The prediction is {0}'.format(id_to_language[prediction]))
  print("Enter some starter text")

