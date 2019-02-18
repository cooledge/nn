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

def files_in_dir(path):
  return [join(path, f) for f in listdir(path)]

vocab_size = 20000
max_len = 10
ngrams = 3 # trigrams
percent_training = 0.8
percent_test = 0.1
percent_validification = 0.1
path = 'data'

languages = [f for f in listdir(path)]
id_to_language = languages
language_to_id =  { language:index for (index, language) in enumerate(id_to_language) }
n_languages = len(id_to_language)

language_dirs = files_in_dir(path)

def word_to_ngrams(word):
  return [word[i:i+ngrams] for i in range(int(len(word)-ngrams+1))]

def get_lines(filename, lines_of_ngrams):
  with open(filename) as f :
    lines = f.readlines()
    for line in lines:
      ngrams = []
      for word in line.lower().split():
        if word.isalpha():
          for ngram in word_to_ngrams(word):
            ngrams.append(ngram)
      if len(ngrams) > 0:
        lines_of_ngrams.append(ngrams)

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

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(sample_text, sample_language, epochs=50, batch_size=100, verbose=1)

counter = 0
for text in sample_text:
  if sum(text) == 0:
    counter += 1
print(counter)

predictions = model.predict(sample_text)
pdb.set_trace()
for prediction, text, language in zip(predictions, sample_text, sample_language):
  if np.argmax(prediction) != language:
    pdb.set_trace()
    print(tokenizer.sequences_to_texts([text]))


pdb.set_trace()
pdb.set_trace()

