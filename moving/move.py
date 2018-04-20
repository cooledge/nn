import tensorflow as tf
import numpy as np
import cv2
import time
import math
import pdb
import time
import urllib
import os.path
import random
import argparse
from chw2hwc import chw2hwc
from PIL import Image

batch_size = 64
nb_epochs = 500
model_dir = './model'

if not os.path.exists(model_dir):
  os.makedirs(model_dir)

model_filename = model_dir + "/model"

INPUT_DIM = 64
data_dir = './data'
files = os.listdir('./data')
# number of files per input
n_files = 3

def filepath(filename):
  return "{0}/{1}".format(data_dir, filename)

def get_counter_fn(filename):
  return int(filename[filename.find('-')+1:filename.find('.')])

def get_counter(i):
  if i >= len(files):
    return -1
  filename = files[i]
  return get_counter_fn(filename)

def get_prefix_fn(filename):
  return filename[:filename.find('-')]

def get_prefix(i):
  if i >= len(files):
    return None
  filename = files[i]
  return get_prefix_fn(filename)

def sort_key(filename):
  return (get_prefix_fn(filename), get_counter_fn(filename))

files = sorted(files, key=sort_key)

def si(image):
  cv2.imshow("", image)
  cv2.waitKey(1)

def load_image(filename):
  try:
    image = Image.open(filepath(filename)).resize((INPUT_DIM, INPUT_DIM))
    image = np.array(image)
    image = np.mean(image, -1)
    return image / 255.
  except IOError:
    return None

def get_category(i):
  file = files[i]
  return file[:file.find('_')] 

def get_image(i):
  if get_counter(i) == get_counter(i+n_files-1) - (n_files-1):
    images = [load_image(files[i+j]) for j in range(n_files)]
    return images
 
categories = ['forward', 'backward', 'left', 'right', 'stop']

def cat2ind(category):
  return categories.index(category)

assert cat2ind(categories[0]) == 0

def ind2cat(index):
  return categories[index]

assert ind2cat(0) == categories[0]

def cats2inds(categories):
  indexes = []
  for category in categories:
    indexes.append(cat2ind(category))
  return indexes

assert cats2inds(categories) == [0,1,2,3,4]

def load_training_data():
  # take N images in a row and make them each a channel in a single
  images = []
  directions = []
  for i in range(len(files)):
    image = get_image(i)
    if image:
      images.append(image)
      directions.append(get_category(i))
    else:
      get_image(i)

  split_at = int(len(images)*0.9)
  X_train = np.array(images[:split_at])
  Y_train = directions[:split_at]

  X_test = np.array(images[split_at:])
  Y_test = directions[split_at:]

  return X_train, Y_train, X_test, Y_test

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# number of classes

img_rows = INPUT_DIM
img_cols = INPUT_DIM

def tf_add_conv(inputs, filters, kernel_size=[5,5], include_pool=True):
  layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding='same', activation=tf.nn.relu)
  if include_pool:
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  return layer

model_input = tf.placeholder(tf.float32, shape=(None, n_files, img_rows, img_cols), name='model_input')
layer = chw2hwc(model_input)
layer = tf_add_conv(model_input, nb_filters, include_pool=False)
layer = tf_add_conv(layer, nb_filters, include_pool=False)
layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
layer = tf.nn.dropout(layer, keep_prob=0.25)
layer = tf.layers.flatten(layer)
layer = tf.layers.dense(layer, 128, activation=tf.nn.relu)
layer = tf.nn.dropout(layer, keep_prob=0.25)
layer = tf.layers.dense(layer, len(categories))
model_logits = layer

model_softmax = tf.nn.softmax(model_logits)
model_predict = tf.argmax(model_softmax, axis=1)
model_output = tf.placeholder(tf.int32, shape=[None], name="model_output")
model_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=model_output, logits=model_logits))
model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op = model_optimizer.minimize(model_loss)
model_accuracy = tf.metrics.accuracy(model_output, model_predict)


session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

saver = tf.train.Saver()

try:
  saver.restore(session, model_filename)
except Exception as e:
  0 # ignore 

class Predict:

  def __init__(self):
    self.images = []

  def run(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    self.images.append(gray)
    if len(self.images) == n_files: 
      predict = session.run(model_predict, { model_input: [self.images] } )
      self.images.pop(0)
      return ind2cat(predict[0])
    return None
    
if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = load_training_data()
  nb_train_samples, img_channels, img_rows, img_cols = X_train.shape
  nb_test_samples = X_test.shape[0]

  print(nb_train_samples, 'train samples')
  print(nb_test_samples, 'test samples')

  for epoch in range(nb_epochs):
    n_batches = nb_train_samples // batch_size

    for batch_no in range(n_batches):
      start = batch_no * batch_size
      end = start + batch_size
      batch_input = X_train[start:end]
      batch_output = cats2inds(Y_train[start:end])

      loss, _, accuracy = session.run([model_loss, model_train_op, model_accuracy], { model_input: batch_input, model_output: batch_output })
      #print("Train Loss {0}, Accuracy: {1}".format(loss, accuracy))
    
    accuracy = session.run(model_accuracy, { model_input: X_train, model_output: cats2inds(Y_train) })
    print("Epoch {0} Test Accuracy: {1}".format(epoch, accuracy))

  saver.save(session, model_filename)

