import tensorflow as tf
from tensorflow import keras
from collections import Counter
import numpy as np
import sys
import re
import pdb

vocab_size = 1000
max_len = 50
filename = 'data/questions.txt'
use_noise = True
noise_filename = 'data/warandpeace.txt'
n_noise = 1000

def load_noise():
  questions = []
  with open(noise_filename) as f:
    questions = [line for line in f.readlines()]
    questions = [question for question in questions if question.strip()]
  return questions[0:n_noise]

noise_questions = load_noise()
noise_answers = ['<NA>' for _ in range(len(noise_questions))]
  
with open(filename) as f:
  lines = f.readlines()
 
lines = [line.split(",") for line in lines]

answerers = [answerers for (answerers, question) in lines]
id_to_answerer = list(set(answerers+['<NA>']))
answerer_to_id = { answerer:index for (index, answerer) in enumerate(id_to_answerer) }
n_answerers = len(set(answerer_to_id))

questions = [question.rstrip() for (answerer, question) in lines]
answerers = [ answerer_to_id[a] for a in answerers]

tokenizer = keras.preprocessing.text.Tokenizer(vocab_size)
tokenizer.fit_on_texts(questions)

def to_sequences(questions):
  questions = tokenizer.texts_to_sequences(questions)
  questions = keras.preprocessing.sequence.pad_sequences(questions, padding='post', maxlen=max_len)
  return questions

questions = to_sequences(questions)
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(n_answerers, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
pdb.set_trace()
model.fit(questions, answerers, epochs=200, batch_size=2, verbose=1)

while True:
  print('Enter a question')
  question = sys.stdin.readline()
  if question.isspace():
    break
  questions = to_sequences([question])
  prediction = model.predict(questions)[0]
  prediction = np.argmax(prediction)
  print(id_to_answerer[prediction])

