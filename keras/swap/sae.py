import tensorflow as tf
import numpy as np
import cv2
import time
import pdb
import os.path
import random
from PIL import Image
from PixelShuffler import PixelShuffler
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam

'''
   increase encoding layer size -> no diff
   add more selector layers between later layers -> got worse
   distor input images -> iffy
   increase the batch size again -> got worse
   train in batches of just trump and cage
   try black and white

   change the font
   men to women
   add a smile
'''

INPUT_DIM = 64
INPUT_SIZE=(INPUT_DIM, INPUT_DIM, 3)
ENCODER_DIM = 1024
BATCH_SIZE = 256
SAVE_DIR = 'models'
SAVE_FILE = 'models/swap'

'''
def warp_image(image):
  #image = cv2.resize(image, (256,256))
  image = random_transform( image, **self.random_transform_args )
  warped_img, target_img = self.random_warp( image, coverage )

  return warped_img, target_img
'''

def random_transform(image, rotation_range=10, zoom_range=0.05, shift_range=0.05, random_flip=0.4):
  h, w = image.shape[0:2]
  rotation = np.random.uniform(-rotation_range, rotation_range)
  scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
  tx = np.random.uniform(-shift_range, shift_range) * w
  ty = np.random.uniform(-shift_range, shift_range) * h
  mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
  mat[:, 2] += (tx, ty)
  result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
  if np.random.random() < random_flip:
    result = result[:, ::-1]
  return result

'''
# get pair of random warped images from aligned face image
def random_warp(image, coverage=160):
  assert image.shape == (256, 256, 3)
  range_ = np.linspace(128 - coverage//2, 128 + coverage//2, 5)
  mapx = np.broadcast_to(range_, (5, 5))
  mapy = mapx.T

  mapx = mapx + np.random.normal(size=(5, 5), scale=5)
  mapy = mapy + np.random.normal(size=(5, 5), scale=5)

  interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
  interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

  warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

  src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
  dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
  mat = umeyama(src_points, dst_points, True)[0:2]

  target_image = cv2.warpAffine(image, mat, (64, 64))
  return warped_image, target_image
''' 

def load_image(filepath):
  try:
    image = Image.open(filepath).resize((INPUT_DIM, INPUT_DIM))
    image = np.array(image)
    return image / 255.
  except IOError:
    return None
  
def load_images2(dir):
  files = os.listdir(dir)
  images = []
  for file in files:
    image = load_image(dir+"/"+file)
    if image is not None:
      np.array(images.append(image))
  return np.array(images)

def load_images(dir):
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    horizontal_flip=True)

  test_datagen = ImageDataGenerator(rescale=1./255)
    
  train_generator = train_datagen.flow_from_directory(
        dir+"/train", 
        target_size=(INPUT_DIM,INPUT_DIM), 
        batch_size=BATCH_SIZE, 
        class_mode='input')

  validation_generator = train_datagen.flow_from_directory(
        dir+"/validation", 
        target_size=(INPUT_DIM,INPUT_DIM), 
        batch_size=BATCH_SIZE, 
        class_mode='input')

  return train_generator, validation_generator

def copy_model(model):
  copy = Sequential()
  for layer in model.layers:
    copy.add(layer)
  return copy

def tf_add_conv(inputs, filters):
  layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
  layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  return layer

def interleave_flat_layer(signals, selectors):
  model_split_a = tf.split(selectors, selectors.shape[1], axis=1)
  model_split_b = tf.split(signals, signals.shape[1], axis=1)

  model_interleaved = []
  for column in model_split_b:
    model_interleaved.append(column)
    model_interleaved.extend(model_split_a)

  model_concat = tf.concat(model_interleaved, 1)
                        
  model_concat = tf.reshape(model_concat, [-1, len(model_interleaved), 1])
  #model_output = tf.layers.conv1d(inputs=model_concat, filters=1, kernel_size=3, strides=3, padding='same', activation=tf.nn.relu)
  #model_output = tf.layers.conv1d(inputs=model_concat, filters=1, kernel_size=3, strides=3, padding='same', activation=tf.nn.sigmoid)
  model_output = tf.layers.conv1d(inputs=model_concat, filters=1, kernel_size=3, strides=3, padding='same')
  model_output = tf.layers.flatten(model_output)
  return model_output

def interleave_layer(signals, selectors):
  flat_signals = tf.layers.flatten(signals)
  selector_layer = interleave_flat_layer(flat_signals, selectors)
  print("Flat size is {0}".format(flat_signals.shape))
  def fixup(dim):
    if dim is None:
      return -1
    else:
      return dim
  shape = [fixup(dim) for dim in signals.shape.as_list()]
  shaped_signals = tf.reshape(selector_layer, shape)
  return shaped_signals

def AutoEncoder():
  model = Sequential()

  model_selector_input = tf.placeholder(tf.float32, shape=[None, 2], name="selectors_input")

  model_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="signal_input")
  layer = model_input

  # Encoder
  layer = tf_add_conv(layer, 128)
  layer = tf_add_conv(layer, 256)
  layer = tf_add_conv(layer, 512)
 
  layer = tf.layers.flatten(layer)
  layer = tf.layers.dense(layer, ENCODER_DIM)
  layer = interleave_flat_layer(layer, model_selector_input)
  layer = tf.layers.dense(layer, 4*4*1024)
  print("Layer 0 size would be {0}".format(tf.layers.flatten(layer).shape[1]))  
  layer = interleave_flat_layer(layer, model_selector_input)
  layer = tf.reshape(layer, [-1,4,4,1024])
  layer = tf.image.resize_images(layer, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  def upsample(inputs, filters, out_size):
    layer = tf.layers.conv2d_transpose(inputs, filters, (5,5), padding='same', activation=tf.nn.relu)
    return tf.image.resize_images(layer, size=(out_size,out_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  print("Layer 1 size would be {0}".format(tf.layers.flatten(layer).shape[1]))  
  #layer = interleave_layer(layer, model_selector_input)
  layer = upsample(layer, 512, 16)
  print("Layer 2 size would be {0}".format(tf.layers.flatten(layer).shape[1]))  
  layer = upsample(layer, 512, 32)
  print("Layer 3 size would be {0}".format(tf.layers.flatten(layer).shape[1]))  
  layer = upsample(layer, 512, 64)
  print("Layer 4 size would be {0}".format(tf.layers.flatten(layer).shape[1]))  
  layer = tf.layers.conv2d_transpose(layer, 3, (5,5), activation=tf.sigmoid, padding='same')

  return layer, model_input, model_selector_input

layer, model_input, model_selector_input = AutoEncoder()

trump_images = load_images2("./photo/trump")
cage_images = load_images2("./photo/cage")

def to_selectors(one_hot, images):
  return np.array([one_hot for _ in images])

# 0 bit is cage, 1 bit is trump
cage_selectors = to_selectors([1,0], cage_images)
trump_selectors = to_selectors([0,1], trump_images)

import matplotlib.pyplot as plt
plt.ion() 
plt.show()

n_rows = 6
n_cols = 4
plt.figure(figsize=(n_rows,n_cols))
n_images = 4

def show_graph(sess):
  def predict(selector, images):
    placeholders ={ 
        model_input: images, 
        model_selector_input: to_selectors(selector, images) 
        }
    return sess.run(layer, placeholders)

  raw_cage = cage_images[0:4]
  decoded_cage = predict([1,0], cage_images[0:4])
  decoded_cage_as_trump = predict([0,1], cage_images[0:4])

  raw_trump= trump_images[0:4]
  decoded_trump = predict([0,1], trump_images[0:4])
  decoded_trump_as_cage = predict([1,0], trump_images[0:4])

  def no_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  def plot_image(col, image):
    ax = plt.subplot(n_rows,n_cols,(4*col)+i+1)
    plt.imshow(image) 
    no_axis(ax)

  for i in range(n_images):
    plot_image(0, raw_cage[i])
    plot_image(1, decoded_cage[i])
    plot_image(2, decoded_cage_as_trump[i])
    plot_image(3, raw_trump[i])
    plot_image(4, decoded_trump[i])
    plot_image(5, decoded_trump_as_cage[i])

  plt.pause(0.001)


saver = tf.train.Saver()

model_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(tf.layers.flatten(model_input), tf.layers.flatten(layer)))
optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op = model_optimizer.minimize(model_loss)

combine_images = False

if combine_images:
  images = np.concatenate((cage_images, trump_images))
  selectors = np.concatenate((cage_selectors, trump_selectors))
  indexes = [i for i in range(len(images))]
else:
  cage_indexes = [i for i in range(len(cage_images))]
  trump_indexes = [i for i in range(len(trump_images))]

#images = cage_images
#selectors = cage_selectors

def get_batch(indexes, start, end, x):
  value = [x[indexes[i]] for i in range(start, end)]
  return value

epochs = 10000
steps = 50
if combine_images:
  batches = len(images) // BATCH_SIZE
else:
  batches = min(len(trump_images), len(cage_images)) // BATCH_SIZE

sess = tf.Session()

sess.run(tf.global_variables_initializer())
saved_model_path = tf.train.latest_checkpoint(SAVE_DIR)
if saved_model_path:
  saver.restore(sess, saved_model_path)

last_time = time.time()
if combine_images:
  for epoch in range(epochs):
    print("\nEpoch {0} seconds {1}".format(epoch, time.time()-last_time))
    last_time = time.time()
    random.shuffle(indexes)
    timages = [random_transform(image) for image in images]
    for step in range(steps):
      for batch in range(batches):
        start = batch*BATCH_SIZE
        end = (batch+1)*BATCH_SIZE
        placeholders = {
          model_input: get_batch(indexes, start, end, timages),
          model_selector_input: get_batch(indexes, start, end, selectors)
        }
        loss, _ = sess.run([model_loss, model_train_op], placeholders)
      print("Step: {0} loss: {1}\r".format(step, loss))
      show_graph(sess)
      saver.save(sess, SAVE_FILE)
else:
  for epoch in range(epochs):
    print("\nEpoch {0} seconds {1}".format(epoch, time.time()-last_time))
    last_time = time.time()
    random.shuffle(cage_indexes)
    random.shuffle(trump_indexes)
    tcage_images = [random_transform(image) for image in cage_images]
    ttrump_images = [random_transform(image) for image in trump_images]
    for step in range(steps):
      for batch in range(batches):
        start = batch*BATCH_SIZE
        end = (batch+1)*BATCH_SIZE

        placeholders = {
          model_input: get_batch(cage_indexes, start, end, tcage_images),
          model_selector_input: get_batch(cage_indexes, start, end, cage_selectors)
        }
        loss, _ = sess.run([model_loss, model_train_op], placeholders)

        placeholders = {
          model_input: get_batch(trump_indexes, start, end, ttrump_images),
          model_selector_input: get_batch(trump_indexes, start, end, trump_selectors)
        }
        loss, _ = sess.run([model_loss, model_train_op], placeholders)
      print("Step: {0} loss: {1}\r".format(step, loss))
      show_graph(sess)
      saver.save(sess, SAVE_FILE)
