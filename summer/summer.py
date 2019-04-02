import tensorflow as tf
from tensorflow import keras
import numpy as np
import pdb

# Input: 2 Numbers and Op (Add, Subtract, Multiply, Divide)
# Output: the result of applying Op


'''
idea: neural net A has all of its nodes multiplied by the output node of another net. zero turns it off. use that as a way to combine
'''

def to_one_hot(n, v):
  oh = [0 for _ in range(n)]
  oh[v] = 1
  return oh

x = []
y = []
for i in range(0, 10):
  for j in range(0, 10):
    x += [to_one_hot(10, i) + to_one_hot(10, j)]
    y += [to_one_hot(100, i+j)]

x = np.array(x)
y = np.array(y)

model = keras.Sequential()
model.add(keras.layers.Dense(4000, input_shape=(10+10,)))
model.add(keras.layers.Dense(100, activation=tf.nn.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(x), np.array(y), epochs=500)
