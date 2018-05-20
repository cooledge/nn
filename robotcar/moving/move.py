import tensorflow as tf
import numpy as np
import cv2
import time
import math
import pdb
import fnmatch
import time
import urllib
import os.path
import random
import argparse
from collections import defaultdict
from collections import deque
# from moving.chw2hwc import chw2hwc
from PIL import Image
import argparse
import glob
import sys

parent_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(parent_dir)
import utils

'''
Epoch 499 Validation Accuracy: 0.9857142857142858 Loss 0.006371537689119577
Test Accuracy: 0.9708333333333333
  forward -> 0.9375
      forward -> 45
      backward -> 1
      stop -> 2
  backward -> 1.0
      backward -> 48
  left -> 0.9166666666666666
      left -> 44
      right -> 4
  right -> 1.0
      right -> 48
  stop -> 1.0
      stop -> 48
'''

batch_size = 64
model_dir = os.path.dirname(os.path.abspath(__file__)) + "/model"

if not os.path.exists(model_dir):
  os.makedirs(model_dir)

model_filename = model_dir + "/model"

INPUT_DIM = 64
n_rows = INPUT_DIM
n_cols = INPUT_DIM

#data_dir = './data'

def get_prefix_fn(filename):
  return filename[:filename.find('-')]

def get_counter_fn(filename):
  return int(filename[filename.find('-')+1:filename.find('.')])

def sort_key(filename):
  return (get_prefix_fn(filename), get_counter_fn(filename))

def get_files(dir):
  files = [file for file in os.listdir(dir) if fnmatch.fnmatch(file, '*.jpg')]
  files = sorted(files, key=sort_key)
  return files

# number of files per input
n_files = 5

def filepath(data_dir, filename):
  return "{0}/{1}".format(data_dir, filename)

def get_counter(files, i):
  if i >= len(files):
    return -1
  filename = files[i]
  return get_counter_fn(filename)

def si(image):
  cv2.imshow("", image)
  cv2.waitKey(1)

'''
def load_image(data_dir, filename):
  try:
    image = Image.open(filepath(data_dir, filename)).resize((INPUT_DIM, INPUT_DIM))
    image = np.array(image)
    image = np.mean(image, -1)
    return image / 255.
  except IOError:
    return None
 '''
def load_image(data_dir, filename):
  utils.load_image(data_dir, filename, INPUT_DIM, INPUT_DIM)

def get_category(files, i):
  file = files[i]
  return file[:file.find('_')] 

def get_image(data_dir, files, i):
  if get_counter(files, i) > 1 and get_counter(files, i) == get_counter(files, i+n_files-1) - (n_files-1):
    images = [load_image(data_dir, files[i+j]) for j in range(n_files)]
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

def load_test_data():
  data_dir = './data/test'
  files = get_files(data_dir)

  # take N images in a row and make them each a channel in a single
  images = []
  directions = []
  for i in range(len(files)):
    image = get_image(data_dir, files, i)
    if image:
      images.append(image)
      directions.append(get_category(files, i))
    else:
      get_image(data_dir, files, i)

  return images, directions

def load_training_data():
  data_dir = './data'
  files = get_files(data_dir)

  # take N images in a row and make them each a channel in a single
  images = []
  directions = []
  for i in range(len(files)):
    image = get_image(data_dir, files, i)
    if image:
      images.append(image)
      directions.append(get_category(files, i))
    else:
      get_image(data_dir, files, i)

  indexes = [i for i in range(len(images))]
  random.shuffle(indexes)
  
  simages = [images[indexes[i]] for i in range(len(images))]
  sdirections = [directions[indexes[i]] for i in range(len(directions))] 

  images = simages
  directions = sdirections

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

# n_files = 3
#   keep_prob = 0.5
#     61%
#     dense = 256
#       58%
#     another conv layer
#       63%
#     [3,3] - max pool
#       58%
#   keep_prob = 0.25
#     59%
# n_files = 4 keep_prob = 0.5
#   65%
#   from 1500 to 3000 test images
#     77%
# n_files == 5
#   66%

model_input = tf.placeholder(tf.float32, shape=(None, n_files, img_rows, img_cols), name='model_input')
model_keep_prob = tf.Variable(0.50, dtype=tf.float32)

def Model2D(model_input):
  #layer = chw2hwc(model_input)
  layer = tf_add_conv(model_input, nb_filters, include_pool=False)
  layer = tf_add_conv(layer, nb_filters, include_pool=False)
  layer = tf_add_conv(layer, nb_filters, include_pool=False)
  layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  #layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[5,5], strides=2, padding='same')
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.flatten(layer)
  layer = tf.layers.dense(layer, 128, activation=tf.nn.relu)
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.dense(layer, len(categories))
  return layer

def Model3D(model_input):
  layer = tf.reshape(model_input, (-1, n_files, img_rows, img_cols, 1))
  layer = tf.layers.conv3d(inputs=layer, filters=nb_filters, kernel_size=n_files, padding='same', activation=tf.nn.relu)
  layer = tf.layers.conv3d(inputs=layer, filters=nb_filters, kernel_size=n_files, padding='same', activation=tf.nn.relu)
  layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[4,2,2], strides=2, padding='same')
  layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[4,4,4], strides=2, padding='same')
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.flatten(layer)
  layer = tf.layers.dense(layer, 128, activation=tf.nn.relu)
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.dense(layer, len(categories))
  return layer

#model_logits = Model2D(model_input)
model_logits = Model3D(model_input)

model_softmax = tf.nn.softmax(model_logits)
model_predict = tf.argmax(model_softmax, axis=1)
model_output = tf.placeholder(tf.int32, shape=[None], name="model_output")
model_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=model_output, logits=model_logits))
model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
#model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-6, beta1=0.5, beta2=0.999)
model_train_op = model_optimizer.minimize(model_loss)
model_accuracy = tf.metrics.accuracy(model_output, model_predict)

session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

saver = tf.train.Saver()

class Predict:

  def __init__(self):
    self.images = deque()

  def run(self, image, display_image):
    from arrow import draw_arrow
    image = cv2.resize(image, (n_cols, n_rows))
    self.images.append(image)
    category = None
    if len(self.images) == n_files: 
      try:
        predict = session.run(model_predict, { model_input: [self.images], model_keep_prob: 1.0 } )
        category = ind2cat(predict[0])
        draw_arrow(display_image, category)
      except ValueError:
        category = None
      self.images.popleft()

    return category

if not __name__ == '__main__':
  try:
    saver.restore(session, model_filename)
  except Exception as e:
    0 # ignore 

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Detect direction of motion")

  parser.add_argument("--epochs", default=500, type=int)
  parser.add_argument("--train", dest='train', default=True, action='store_true')
  parser.add_argument("--no-train", dest='train', action='store_false')
  parser.add_argument("--test", dest='test', default=True, action='store_true')
  parser.add_argument("--no-test", dest='test', action='store_false')
  parser.add_argument("--clean", dest='clean', default=False, action='store_true')
  parser.add_argument("--test-predict", dest='test_predict', default=False, type=bool)

  args = parser.parse_args()

  if not args.clean:
    try:
      saver.restore(session, model_filename)
    except Exception as e:
      0 # ignore 

  if args.clean:
    os.system("rm {0}/*".format(model_dir))

  nb_epochs = args.epochs

  X_train, Y_train, X_validation, Y_validation = load_training_data()

  if args.train:
    nb_train_samples, img_channels, img_rows, img_cols = X_train.shape
    nb_validation_samples = X_validation.shape[0]

    print(nb_train_samples, 'train samples')
    print(nb_validation_samples, 'validation samples')

    for epoch in range(nb_epochs):
      n_batches = nb_train_samples // batch_size

      for batch_no in range(n_batches):
        start = batch_no * batch_size
        end = start + batch_size
        batch_input = X_train[start:end]
        batch_output = cats2inds(Y_train[start:end])

        loss, _ = session.run([model_loss, model_train_op], { model_input: batch_input, model_output: batch_output })
        #print("Train Loss {0}".format(loss))
      
      labels, predictions = session.run([model_output, model_predict], { model_input: X_validation, model_output: cats2inds(Y_validation), model_keep_prob: 1.0 })
      right = len([True for (label, prediction) in zip(labels, predictions) if label == prediction])
      accuracy = float(right) / float(len(predictions)) 
      print("Epoch {0} Validation Accuracy: {1} Loss {2}".format(epoch, accuracy, loss))
      #  accuracy = session.run(model_accuracy, { model_input: X_train, model_output: cats2inds(Y_train) })
      #  print("Epoch {0} Batch Accuracy: {1}".format(epoch, accuracy))

  if args.test:
    X_test, Y_test = load_test_data()
    labels, predictions = session.run([model_output, model_predict], { model_input: X_test, model_output: cats2inds(Y_test), model_keep_prob: 1.0 })
    right = len([True for (label, prediction) in zip(labels, predictions) if label == prediction])
    category_to_right = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0 }
    category_to_other = {}
    category_to_total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0 }
    for label, prediction in zip(labels, predictions):
      if label == prediction:
        category_to_right[label] += 1
      if not label in category_to_other:
        category_to_other[label] = {}
      ci = category_to_other[label]
      if not prediction in ci:
        ci[prediction] = 0
      ci[prediction] += 1
       
      category_to_total[label] += 1

    accuracy = float(right) / float(len(predictions)) 
    print("Test Accuracy: {0}".format(accuracy))
    for category in range(len(categories)):
      print("\t{0} -> {1}".format(categories[category], category_to_right[category]/category_to_total[category]))
      cto = category_to_other[category]
      for key in cto.keys():
        print("\t\t{0} -> {1}".format(categories[key], cto[key]))

  saver.save(session, model_filename)

