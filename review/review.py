import tensorflow as tf
from tensorflow import keras
from collections import Counter
import re
import pdb

vocab_size = 100
max_len = 50
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
tokenizer = keras.preprocessing.text.Tokenizer(vocab_size)
tokenizer.fit_on_texts(questions)
questions = tokenizer.texts_to_sequences(questions)
questions = keras.preprocessing.sequence.pad_sequences(questions, padding='post', maxlen=max_len)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(n_answerers, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
pdb.set_trace()
model.fit(questions, answerers, epochs=200, batch_size=2, verbose=1)
