
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import pdb

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels_sparse), (test_images, test_labels_sparse) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

n_classes = len(class_names)

test_labels = []
for label in test_labels_sparse:
  one_hot = [0.]*n_classes
  one_hot[label] = 1.0
  test_labels.append( one_hot )
test_labels = np.array(test_labels)

train_labels = []
for label in train_labels_sparse:
  one_hot = [0.0]*n_classes
  one_hot[label] = 1.0
  train_labels.append( one_hot )
train_labels = np.array(train_labels)

inputs = keras.layers.Input(shape=(28,28))
flatten = keras.layers.Flatten()(inputs)

l1 = keras.layers.Dense(128, activation=tf.nn.relu)
poutput = l1(flatten)
l2 = keras.layers.Dense(10, activation=tf.nn.softmax, name='positive')
poutput = l2(poutput)
pmodel=keras.Model(inputs=inputs, outputs=poutput)

il1 = keras.layers.Dense(128, activation=tf.nn.relu, trainable=False)
ioutput = il1(flatten)
il1.set_weights(l1.get_weights())
il2 = keras.layers.Dense(10, activation=tf.nn.softmax, name='inverted')
ioutput = il2(ioutput)
il2.set_weights(l2.get_weights())
imodel=keras.Model(inputs=inputs, outputs=ioutput)

pmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
pmodel.fit([train_images], [train_labels], epochs = 10);

# get wrong answers

# inverse train? 

def get_itrain_data(images, labels):
  predictions = pmodel.predict([images])

  itrain_images = []
  itrain_labels = []
  for i in range(len(predictions)):
    pred = np.argmax(predictions[i])
    label = np.argmax(labels[i])
    if pred != label:
      itrain_images.append(images[0])
      itrain_labels.append(pred)

  itrain_images = np.array(itrain_images)
  itrain_labels = np.array(itrain_labels)
  return itrain_images, itrain_labels

itrain_images, itrain_labels = get_itrain_data(test_images, test_labels)

def inverse_loss(y_true, y_pred, from_logits=False, axis=-1):
  #return -keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)
  loss = keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)
  return tf.math.divide(tf.constant(1.0), loss+0.0000001)

test_loss, test_acc = pmodel.evaluate(test_images, test_labels)
print('Before Test accuracy:', test_acc)
print('There are {0} inverse images'.format(len(itrain_images)))

for i in range(1):
  imodel.compile(optimizer='adam', loss=inverse_loss, metrics=['accuracy'])
  imodel.fit([itrain_images], [itrain_labels], epochs = 10);

  #pmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  #pmodel.fit([train_images], [train_labels], epochs = 10);

test_loss, test_acc = pmodel.evaluate(test_images, test_labels)
print('After Test accuracy:', test_acc)
