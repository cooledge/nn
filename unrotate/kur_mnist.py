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

INPUT_DIM = 28
INPUT_SIZE=(INPUT_DIM, INPUT_DIM)
ENCODER_DIM = 1024
BATCH_SIZE = 32
SAVE_DIR = 'models'
SAVE_FILE = 'models/swap'
DATA_DIR = 'photo/data'

data_dir = DATA_DIR
prefix = 'vid'

def si(image):
  cv2.imshow("", image)
  cv2.waitKey(1000)

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

def load_image(filepath):
  try:
    image = Image.open(filepath).resize((INPUT_DIM, INPUT_DIM))
    image = np.array(image)
    return image / 255.
  except IOError:
    return None
  
def load_images(dir):
  files = os.listdir(dir)
  images = []
  for file in files:
    image = load_image(dir+"/"+file)
    if image is not None:
      np.array(images.append(image))
  return np.array(images)

def tf_add_conv(inputs, filters, kernel_size=[5,5], include_pool=True):
  layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding='same', activation=tf.nn.relu)
  if include_pool:
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  return layer

def number_of_rotates():
  return 360

def rotate_to_one_hot_index(degrees):
  return round(degrees)

def one_hot_index_to_rotate(index):
  return index

def rotate_to_one_hot(degrees):
  index = rotate_to_one_hot_index(degrees)
  return [1 if i == index else 0 for i in range(number_of_rotates())]  

def one_hot_to_rotate(one_hot):
  return one_hot_index_to_rotate(np.argmax(one_hot))

def Categorizer(layer):
  layer = tf.nn.relu(layer)
  return tf.layers.dense(layer, number_of_rotates())

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

# model definition
input = Input(shape=(img_rows, img_cols, 1))
x = Reshape((-1, img_rows, img_cols, img_channels))(input)
x = Conv2D(nb_filters, kernel_size, activation='relu')(input)
x = Conv2D(nb_filters, kernel_size, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(nb_classes, activation='softmax')(x)

model = Model(inputs=input, outputs=x)

model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[utils.angle_error])

# training parameters
batch_size = 128
nb_epoch = 50


if False:
  model.fit_generator(
      utils.RotNetDataGenerator(
        X_train,
        batch_size=batch_size,
        preprocess_func=utils.binarize_images,
        shuffle=True
      ),
      steps_per_epoch=nb_train_samples / batch_size,
      epochs=nb_epoch,
      validation_data=utils.RotNetDataGenerator(
        X_test,
        batch_size=batch_size,
        preprocess_func=utils.binarize_images
        ),
      validation_steps=nb_test_samples / batch_size,
      verbose=1,
    )
elif True:
  for epoch in range(50):
    n_batches = nb_train_samples // batch_size
    X_train_rotated, rotated_angles = rotate_images(X_train)
    for batch_no in range(n_batches):
      start = batch_no * batch_size
      end = start + batch_size
      model.train_on_batch( X_train_rotated[start:end],
          to_categorical(rotated_angles[start:end], 360),
        )
    X_test_rotated, test_rotated_angles = rotate_images(X_test)
    result = model.test_on_batch(X_test_rotated, to_categorical(test_rotated_angles, 360))
    print("Epoch {0}, Loss {1}, Angle Diff: {2}".format(epoch, result[0], result[1]))

else:
  X_train_rotated, rotated_angles = rotate_images(X_train)
  model.fit( X_train_rotated,
      to_categorical(rotated_angles, 360),
      batch_size = batch_size,
      epochs=50,
      validation_split = 0.1,
      verbose=1,
    )
