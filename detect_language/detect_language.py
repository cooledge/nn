import tensorflow as tf

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

path = 'data'
languages = [f for f in listdir(path)]
language_dirs = files_in_dir(path)

def get_words(filename, words):
  with open(filename) as f :
    lines = f.readlines()
    for line in lines:
      words.update(line.lower().split())

lang_to_words = {}
for language in languages:
  words = set()
  for f in listdir(join(path, language)):
    get_words(join(path, join(language, f)), words)
  lang_to_words[language] = words;

pdb.set_trace()
