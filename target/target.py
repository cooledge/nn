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
from scipy.stats import multivariate_normal

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
n_radius = int(200 / n_scale)
n_rows = int(480 / n_scale)
n_cols = int(640 / n_scale)
n_filters = 64
batch_size = 1

train_percent = 90

def filepath(data_dir, filename):
  return "{0}/{1}".format(data_dir, filename)

# y = (radius, col, row)
def filename_to_y(filename):
  return [int(c)/n_scale for c in filename.split('.')[0].split('_')]

def si(image):
  cv2.imshow("", image)
  cv2.waitKey(10)

def load_image(data_dir, filename):
  try:
    image = cv2.imread(filepath(data_dir, filename), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (n_cols, n_rows))
    return image / 255.
  except IOError:
    return None

# call it with row,column

def max_differentiable(x, y):
  return x/2. + y/2. + abs(x-y)/2.

def min_differentiable(x, y):
  return x/2. + y/2. - abs(x-y)/2.

def show_field(field):
  img = np.zeros((n_rows, n_cols))
  for r in range(n_rows):
    for c in range(n_cols):
      img[r][c] = field[r][c] * 255.
  si(img)

def label_to_normal_field(row, col):
  scale = 1
  nd = multivariate_normal(mean=[row,col], cov=[[n_rows/scale,0],[0,n_cols/scale]])
  field = np.zeros((n_rows, n_cols))
  for r in range(n_rows):
    for c in range(n_cols):
      field[r, c] = nd.pdf([r,c])
  scale = 1.0 / np.sum(field)
  return field*scale

def loss_field(target, prediction):
  return 1.0-np.sum(min_differentiable(target, prediction/np.sum(prediction)))

def tf_loss_field(targets, predictions):
  scale = tf.reduce_sum(predictions, axis=[1,2])
  normalized_predictions = tf.reshape(predictions, (-1, n_rows*n_cols)) / tf.reshape(scale, (-1, 1))
  normalized_predictions = tf.reshape(normalized_predictions, (-1, n_rows, n_cols))
  return 1.0-tf.reduce_sum(min_differentiable(targets, normalized_predictions), axis=[1,2])

'''
session = tf.Session()

#show_field(label_to_normal_field(10, 44))
area1 = label_to_normal_field(46, 22)
tf_area_labels = tf.constant(np.array([area1, area1, area1]))
area2 = label_to_normal_field(20, 20)
area2_4 = label_to_normal_field(20, 20)*4
area3 = label_to_normal_field(30, 30)
tf_area_predictions = tf.constant(np.array([area2, area2_4, area3]))
loss1 = loss_field(area1, area1)
loss12 = loss_field(area1, area2)
loss12_4 = loss_field(area1, area2_4)
loss13 = loss_field(area1, area3)
pdb.set_trace()
xxx = session.run(tf_loss_field(tf_area_labels, tf_area_predictions))
tf_loss12 = session.run(loss_field(tf_area1, tf_area2))
loss2 = loss_field(area2, area2)
loss21 = loss_field(area2, area1)
loss3 = loss_field(area3, area3)
loss31 = loss_field(area3, area1)
pdb.set_trace()
'''

def load_training_data():
  files = os.listdir(data_dir)

  X = []
  Y_radius = []
  Y_position = []

  for filename in files:
    radius, row, col, _bgidx = filename_to_y(filename)
    image = load_image(data_dir, filename)
    Y_radius.append(radius)
    Y_position.append((row,col))
    X.append(image)

  index = [i for i in range(len(X))]
  random.shuffle(index)
  n_train = int(len(X) * train_percent / 100)
  if n_train == 0:
    n_train = 1
  X_train = [X[i] for i in index[:n_train]]
  X_validation = [X[i] for i in index[n_train:]]

  Y_train_radius = [Y_radius[i] for i in index[:n_train]]
  Y_validation_radius = [Y_radius[i] for i in index[n_train:]]

  Y_train_position = [Y_position[i] for i in index[:n_train]]
  Y_validation_position = [Y_position[i] for i in index[n_train:]]

  return np.array(X_train), np.array(Y_train_radius), np.array(Y_train_position), np.array(X_validation), np.array(Y_validation_radius), np.array(Y_validation_position)

model_keep_prob = tf.Variable(1.00, dtype=tf.float32)
model_input = tf.placeholder(tf.float32, shape=(None, n_rows, n_cols))

def tf_add_conv(inputs, filters, kernel_size=[5,5], include_pool=True):
  layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding='same', activation=tf.nn.relu)
  if include_pool:
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  return layer

def NoModel(model_input, output_size):
  layer = tf.reshape(model_input, (-1, n_rows*n_cols))
  return tf.layers.dense(layer, output_size)

def Model2D(model_input, output_size):
  #layer = chw2hwc(model_input)
  layer = tf_add_conv(model_input, n_filters, include_pool=False)
  layer = tf_add_conv(layer, 256, include_pool=False)
  layer = tf_add_conv(layer, 128, include_pool=False)
  layer = tf_add_conv(layer, 64, include_pool=False)
  layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  #layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[5,5], strides=2, padding='same')
  #layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.flatten(layer)
  layer = tf.layers.dense(layer, 512, activation=tf.nn.relu)
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.dense(layer, output_size) 
  layer = tf.nn.relu(layer)
  return layer

layer = Model2D(tf.reshape(model_input, (-1, n_rows, n_cols, 1)), n_radius+n_rows*n_cols)
#layer = NoModel(tf.reshape(model_input, (-1, n_rows, n_cols, 1)), n_radius+n_rows*n_cols)
#layer = tf.layers.dense(layer, n_radius+n_rows*n_cols)
model_logits = layer
model_logits_radius, model_logits_position  = tf.split(model_logits, [n_radius, n_rows*n_cols], 1)
model_predict_radius = tf.argmax(model_logits_radius, axis=1)

def argmax_2d(one_d):
  one_d_argmax = tf.argmax(one_d, axis=1)
  row_argmax = one_d_argmax // n_cols
  col_argmax = one_d_argmax % n_cols
  return tf.stack((row_argmax, col_argmax), axis=1)

model_predict_position = argmax_2d(model_logits_position)
model_logits_position = tf.reshape(model_logits_position, (-1, n_rows, n_cols))
#model_logits_position = tf.reshape(model_logits_position, (-1, n_rows, n_cols, 1))

model_output_radius = tf.placeholder(tf.float32, shape=(None, n_radius))
model_output_position = tf.placeholder(tf.float32, shape=(None, n_rows, n_cols))

def maxpool(layer):
  return tf.layers.max_pooling2d(tf.reshape(layer, (-1, n_rows, n_cols, 1)), 3, 2)

#model_loss = (1.0-tf.reduce_sum(model_output_position*model_logits_position))
model_loss = tf.reduce_sum(tf_loss_field(model_output_position, model_logits_position))# + tf.reduce_sum(model_logits_position)

model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op = model_optimizer.minimize(model_loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

def show_graph(labels, predictions):
  img = np.zeros((n_rows, n_cols))
  cv2.circle(img, (int(labels[0][1]), int(labels[0][0])), 5, (255,0,255), -1)
  cv2.circle(img, (int(predictions[0][1]), int(predictions[0][0])), 5, (255,255,255), -1)
  si(img)

def accuracy(X, Y_position): 
  predictions = session.run(model_predict_position, { model_input: X, model_keep_prob: 1.0 })
  labels = Y_position
  right = len([True for (label, prediction) in zip(labels, predictions) if tuple(label) == tuple(prediction)])
  distance = [ math.pow(math.pow(label[0]-prediction[0], 2) + math.pow(label[1]-prediction[1], 2), 0.5) for (label, prediction) in zip(labels, predictions) ]
  mean_distance = sum(distance) / len(distance)
  accuracy = float(right) / float(len(predictions))
  print("Epoch {0} Validation Accuracy: {1} Mean Distance: {2} Loss {3}".format(epoch, accuracy, mean_distance, loss))
  print("Labels {0} Predictions {1}".format(labels, predictions))
  show_graph(labels, predictions)

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
          one_hot[int(radius)] = 1
          batch_output_radius.append(one_hot)

        batch_output_position = []
        for (col, row) in Y_train_position[start:end]:
          #one_hot = np.zeros((n_rows, n_cols))
          #one_hot[int(row)][int(col)] = 1
          field = label_to_normal_field(row, col)
          #batch_output_position.append(one_hot)
          batch_output_position.append(field)

        placeholders = { 
          model_input: batch_input, 
          model_output_radius: batch_output_radius, 
          model_output_position: batch_output_position 
        }
        loss, mop, mlp, _ = session.run([model_loss, model_output_position, model_logits_position, model_train_op], placeholders)
        if loss == 0 or epoch == 480:
          pdb.set_trace()
          pdb.set_trace()

        #accuracy(batch_input, Y_train_position[start:end])
        #print("Train Loss {0}".format(loss))
    
      #accuracy(X_validation, Y_validation_position)
      accuracy(X_train, Y_train_position)
      #  accuracy = session.run(model_accuracy, { model_input: X_train, model_output: cats2inds(Y_train) })
      #  print("Epoch {0} Batch Accuracy: {1}".format(epoch, accuracy))

