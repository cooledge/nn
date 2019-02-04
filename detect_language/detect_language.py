import tensorflow as tf
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

n_most_common = 1000
n_words = 3
percent_training = 0.8
percent_test = 0.1
percent_validification = 0.1
path = 'data'

def get_lang_to_vocab():
  languages = [f for f in listdir(path)]
  language_dirs = files_in_dir(path)

  def get_words(filename, words):
    with open(filename) as f :
      lines = f.readlines()
      for line in lines:
        words.update([word for word in line.lower().split() if word.isalpha()])

  lang_to_vocab = {}
  for language in languages:
    words = Counter()
    for f in listdir(join(path, language)):
      get_words(join(path, join(language, f)), words)
    lang_to_vocab[language] = [word for (word,_) in words.most_common(n_most_common)];

  return lang_to_vocab

def list_to_maps(values):
  value_to_id = {}
  id_to_value = {}
  for i, value in enumerate(values):
    value_to_id[value] = i
    id_to_value[i] = value
  return (value_to_id, id_to_value)

def get_dicts(lang_to_vocab):
  words = []
  for language in lang_to_vocab.keys():
    words += lang_to_vocab[language]
  return list_to_maps(words)

lang_to_vocab = get_lang_to_vocab()
languages = lang_to_vocab.keys()
lang_to_id, id_to_lang = list_to_maps(languages)
n_languages = len(languages)
word_to_id, id_to_word = get_dicts(lang_to_vocab)

# set up training data - list of words
print(lang_to_vocab)
print(word_to_id)
print(id_to_word)

def get_lang_to_words():
  languages = [f for f in listdir(path)]
  language_dirs = files_in_dir(path)

  def get_words(filename, words):
    with open(filename) as f :
      words += f.read().lower().split()

  lang_to_words = {}
  for language in languages:
    words = []
    for f in listdir(join(path, language)):
      get_words(join(path, join(language, f)), words)
    lang_to_words[language] = words
  return lang_to_words

lang_to_words = get_lang_to_words()
print(lang_to_words)

def get_sample_stats():
  counter = 0
  total = 0
  sample_languages = []
  sample_words = []
  for language in languages:
    words = lang_to_words[language]
    for i in range(int(len(words)/n_words)):
      sample = words[i:i+n_words]
      sample = [word_to_id[w] for w in sample if w in word_to_id]
      pdb.set_trace()
      sample_words.append(sample)
      sample_languages.append(lang_to_id[language])
      total += 1
      if len(sample) > 0:
        counter += 1
  print(counter)
  print(total)
  return samples

samples = get_sample_stats()
print(samples)

'''
samples = samples.shuffle()
training = samples[0:percent_training*len(samples)]
test = samples[0:percent_test*len(samples)]
validification = samples[0:percent_validification*len(samples)]
'''
pdb.set_trace()
model = keras.Sequential()
model.add(keras.layers.GlobalAveragePooling())
model.add(keras.layers.Dense(128), activation=tf.nn.relu)
model.add(keras.layers.Dense(n_languages), activation=tf.nn.sigmoid)

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(questions, answerers, epochs=200, batch_size=2, verbose=1)

