import tensorflow as tf
from tensorflow import keras
from collections import Counter
import re
import pdb

filename = 'data/questions.txt'

with open(filename) as f:
  lines = f.readlines()
 
lines = [line.split(",") for line in lines]

answerer = [answerer for (answerer, question) in lines]
questions = [question.rstrip() for (answerer, question) in lines]
questions = [tf.keras.preprocessing.text.text_to_word_sequence(question) for question in questions]

words = []
[ words.extend(question) for question in questions]
counter = Counter(words)
words = counter.most_common()

id_to_word_map = [word for (word, count) in words]
word_to_id_map = { word:index for (word, index) in enumerate(id_to_word_map) }
pdb.set_trace()
questions = [ [word_to_id_map[word] for word in question] for question in questions ]
pdb.set_trace()
pdb.set_trace()
