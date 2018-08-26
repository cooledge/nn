import tensorflow as tf
from tensorflow import keras
from collections import Counter
import re
import pdb

filename = 'data/questions.txt'

with open(filename) as f:
  lines = f.readlines()
 
lines = [line.split(",") for line in lines]

answerers = [answerers for (answerers, question) in lines]
id_to_answerer = list(set(answerers))
answerer_to_id = { answerer:index for (index, answerer) in enumerate(id_to_answerer) }
answerers = [ answerer_to_id[a] for a in answerers]

n_answerers = len(set(answerers))
questions = [question.rstrip() for (answerer, question) in lines]
questions = [tf.keras.preprocessing.text.text_to_word_sequence(question) for question in questions]

words = []
[ words.extend(question) for question in questions]
counter = Counter(words)
words = counter.most_common()

special = [('<PAD>',0), ('<START>',0), ('<UNK>',0), ('<UNUSED>',0)]
words = special + words
id_to_word_map = [word for (word, count) in words]
word_to_id_map = { word:index+3 for (index, word) in enumerate(id_to_word_map) }
word_to_id_map['<PAD>'] = 0
word_to_id_map['<START>'] = 1
word_to_id_map['<UNK>'] = 2
word_to_id_map['<UNUSED>'] = 3

questions = [ [word_to_id_map[word] for word in question] for question in questions ]
questions = keras.preprocessing.sequence.pad_sequences(questions, value=word_to_id_map['<PAD>'], padding='post', maxlen=50)

vocab_size = len(id_to_word_map)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(n_answerers, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(questions, answerers, epochs=200, batch_size=2, verbose=1)
