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
import utils
from umeyama import umeyama
from PIL import Image

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils.np_utils import to_categorical

def rotate_images(images):
  rotated_images = []
  rotations = []
  for image in images:
    ri, ra = rotate_image(image)
    rotated_images.append(ri)
    rotations.append(ra)

  return np.array(rotated_images).reshape(-1, 28, 28, 1), rotations

def rotate_image(image):
  rotation_angle = np.random.randint(360)
  rotated_image = utils.generate_rotated_image(
                  image,
                  rotation_angle,
                  #size=image.shape[:2],
                  #crop_center=utils.crop_center,
                  #crop_largest_rect=utils.crop_largest_rect
                  )
  return rotated_image, rotation_angle

(X_train,_), (X_test, _) = tf.keras.datasets.mnist.load_data()

model_name = 'rotnet_mnist'

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# number of classes
nb_classes = 360

nb_train_samples, img_rows, img_cols = X_train.shape
img_channels = 1
input_shape = (img_rows, img_cols, img_channels)
nb_test_samples = X_test.shape[0]

print('Input shape:', input_shape)
print(nb_train_samples, 'train samples')
print(nb_test_samples, 'test samples')

def tf_add_conv(inputs, filters, kernel_size=[5,5], include_pool=True):
  layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding='same', activation=tf.nn.relu)
  if include_pool:
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  return layer

model_input = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, 1), name='model_input')
layer = tf.reshape(model_input, (-1, img_rows, img_cols, img_channels))
layer = tf_add_conv(layer, nb_filters, include_pool=False)
layer = tf_add_conv(layer, nb_filters, include_pool=False)
layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
layer = tf.nn.dropout(layer, keep_prob=0.25)
layer = tf.layers.flatten(layer)
layer = tf.layers.dense(layer, 128, activation=tf.nn.relu)
layer = tf.nn.dropout(layer, keep_prob=0.25)
layer = tf.layers.dense(layer, nb_classes)

model_prob_cat = tf.nn.softmax(layer)
model_output = tf.placeholder(tf.int32, shape=[None], name="model_output")
model_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=model_output, logits=layer))
model_optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
model_train_op = model_optimizer.minimize(model_loss)

# training parameters
batch_size = 128
nb_epoch = 50

session = tf.Session()
session.run(tf.global_variables_initializer())
#session.run(tf.local_variables_initializer())

for epoch in range(50):
  n_batches = nb_train_samples // batch_size
  X_train_rotated, rotated_angles = rotate_images(X_train)

  for batch_no in range(n_batches):
    start = batch_no * batch_size
    end = start + batch_size
    batch_input = X_train_rotated[start:end]
    batch_output = rotated_angles[start:end]
    batch_output_cat = to_categorical(rotated_angles[start:end], 360)

    gradients1, loss1 = session.run([model_loss, model_optimizer.compute_gradients(model_loss)], { model_input: batch_input, model_output: batch_output })
    loss_before = session.run(model_loss, { model_input: batch_input, model_output: batch_output })
    loss, _ = session.run([model_loss, model_train_op], { model_input: batch_input, model_output: batch_output })
    loss_after = session.run(model_loss, { model_input: batch_input, model_output: batch_output })
    print("tf loss {0}".format(loss))

  X_test_rotated, test_rotated_angles = rotate_images(X_test)
  result = model.test_on_batch(X_test_rotated, to_categorical(test_rotated_angles, 360))
  print("Epoch {0}, Loss {1}, Angle Diff: {2}".format(epoch, result[0], result[1]))

