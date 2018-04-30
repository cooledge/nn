import cv2
import numpy as np
import sys
import math
import os
import time
import pdb
import argparse
import tensorflow as tf
from PIL import Image
import random

parser = argparse.ArgumentParser(description="Detect direction of motion")

parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--train", dest='train', default=True, action='store_true')
parser.add_argument("--no-train", dest='train', action='store_false')
parser.add_argument("--test", dest='test', default=True, action='store_true')
parser.add_argument("--no-test", dest='test', action='store_false')
parser.add_argument("--clean", dest='clean', default=False, action='store_true')
parser.add_argument("--test-predict", dest='test_predict', default=False, type=bool)

args = parser.parse_args()

data_dir = './data'

# shape (NImages, Cols, Rows)
images = []

n_scale = 5
n_radius = 200 / n_scale
n_rows = 480 / n_scale
n_cols = 640 / n_scale
batch_size = 64

train_percent = 90

def filepath(data_dir, filename):
  return "{0}/{1}".format(data_dir, filename)

# y = (radius, col, row)
def filename_to_y(filename):
  return [int(c)/n_scale for c in filename.split('.')[0].split('_')]

def si(image):
  cv2.imshow("", image)
  cv2.waitKey(1)

def load_image(data_dir, filename):
  try:
    image = cv2.imread(filepath(data_dir, filename), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (n_cols, n_rows))
    return image / 255.
  except IOError:
    return None

def load_training_data():
  files = os.listdir(data_dir)

  X = []
  Y_radius = []
  Y_position = []

  for filename in files:
    radius, col, row, _bgidx = filename_to_y(filename)
    image = load_image(data_dir, filename)
    Y_radius.append(radius)
    Y_position.append((col,row))
    X.append(image)

  index = [i for i in range(len(X))]
  random.shuffle(index)
  n_train = len(X) * train_percent / 100

  X_train = [X[i] for i in index[:n_train]]
  X_validation = [X[i] for i in index[n_train:]]

  Y_train_radius = [Y_radius[i] for i in index[:n_train]]
  Y_validation_radius = [Y_radius[i] for i in index[n_train:]]

  Y_train_position = [Y_position[i] for i in index[:n_train]]
  Y_validation_position = [Y_position[i] for i in index[n_train:]]

  return np.array(X_train), np.array(Y_train_radius), np.array(Y_train_position), np.array(X_validation), np.array(Y_validation_radius), np.array(Y_validation_position)

model_keep_prob = tf.Variable(0.50, dtype=tf.float32)
model_input = tf.placeholder(tf.float32, shape=(None, n_rows, n_cols))
layer = tf.reshape(model_input, (-1, n_rows*n_cols))
layer = tf.layers.dense(layer, n_radius+n_rows*n_cols)
model_logits = layer
model_logits_radius, model_logits_position  = tf.split(model_logits, [n_radius, n_rows*n_cols], 1)
model_predict_radius = tf.argmax(model_logits_radius, axis=1)

def argmax_2d(one_d):
  one_d_argmax = tf.argmax(one_d, axis=1)
  row_argmax = one_d_argmax / n_cols
  col_argmax = one_d_argmax % n_cols
  return tf.stack((row_argmax, col_argmax), axis=1)

model_predict_position = argmax_2d(model_logits_position)
'''
def test():
  pdb.set_trace()
  two_d = [ [ [0,0,0], [0,1,0], [0,0,0], [0,0,0] ], [ [0,0,0], [0,1,0], [0,0,0], [0,0,2] ] ]
  session = tf.Session()
  model_two_d = tf.placeholder(tf.float32, shape=(None, 4, 3))
  placeholders = { model_two_d: two_d }
  result = session.run(argmax_2d(model_two_d), placeholders)
   
test() 
'''

model_logits_position = tf.reshape(model_logits_position, (-1, n_rows, n_cols, 1))

model_output_radius = tf.placeholder(tf.float32, shape=(None, n_radius))
model_output_position = tf.placeholder(tf.float32, shape=(None, n_rows, n_cols))

def maxpool(layer):
  return tf.layers.max_pooling2d(tf.reshape(layer, (-1, n_rows, n_cols, 1)), 3, 2)

model_loss = tf.losses.mean_squared_error(maxpool(model_output_position), maxpool(model_logits_position))
model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op = model_optimizer.minimize(model_loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

if __name__ == "__main__":
  X_train, Y_train_radius, Y_train_position, X_validation, Y_validation_radius, Y_validation_position = load_training_data()

  if args.train:
    nb_train_samples, img_rows, img_cols = X_train.shape
    nb_validation_samples = X_validation.shape[0]

    print(nb_train_samples, 'train samples')
    print(nb_validation_samples, 'validation samples')

    for epoch in range(args.epochs):
      n_batches = nb_train_samples // batch_size

      for batch_no in range(n_batches):
        start = batch_no * batch_size
        end = start + batch_size
        batch_input = X_train[start:end]
        
        batch_output_radius = []
        for radius in Y_train_radius[start:end]:
          one_hot = np.zeros((n_radius))
          one_hot[radius] = 1
          batch_output_radius.append(one_hot)

        batch_output_position = []
        for (col, row) in Y_train_position[start:end]:
          one_hot = np.zeros((n_rows, n_cols))
          one_hot[row][col] = 1
          batch_output_position.append(one_hot)

        placeholders = { 
          model_input: batch_input, 
          model_output_radius: batch_output_radius, 
          model_output_position: batch_output_position 
        }
        loss, _ = session.run([model_loss, model_train_op], placeholders)
        #print("Train Loss {0}".format(loss))
     
      predictions = session.run(model_predict_position, { model_input: X_validation, model_keep_prob: 1.0 })
      labels = Y_validation_position
      right = len([True for (label, prediction) in zip(labels, predictions) if tuple(label) == tuple(prediction)])
      distance = [ math.pow(math.pow(label[0]-prediction[0], 2) + math.pow(label[1]-prediction[1], 2), 0.5) for (label, prediction) in zip(labels, predictions) ]
      mean_distance = sum(distance) / len(distance)
      accuracy = float(right) / float(len(predictions))
      print("Epoch {0} Validation Accuracy: {1} Mean Distance: {2} Loss {3}".format(epoch, accuracy, mean_distance, loss))
      #  accuracy = session.run(model_accuracy, { model_input: X_train, model_output: cats2inds(Y_train) })
      #  print("Epoch {0} Batch Accuracy: {1}".format(epoch, accuracy))
