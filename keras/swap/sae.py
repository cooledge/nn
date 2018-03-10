import tensorflow as tf
import numpy as np
import pdb
import os.path
from PIL import Image
from PixelShuffler import PixelShuffler
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam

INPUT_DIM = 64
INPUT_SIZE=(INPUT_DIM, INPUT_DIM, 3)
ENCODER_DIM = 1024
BATCH_SIZE = 64

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

#def selector_layer(selectors, features):

def tf_add_conv(inputs, filters):
  layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
  layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  return layer

def AutoEncoder():
  model = Sequential()

  layer = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

  # Encoder
  model.add(Conv2D(128, (5,5), activation='relu', padding='same', input_shape=INPUT_SIZE))
  model.add(MaxPooling2D(2,2, padding='same'))
  layer = tf_add_conv(layer, 128)
  model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
  model.add(MaxPooling2D(2,2, padding='same'))
  layer = tf_add_conv(layer, 256)
  model.add(Conv2D(512, (5,5), activation='relu', padding='same'))
  model.add(MaxPooling2D(2,2, padding='same'))
  layer = tf_add_conv(layer, 512)
 
  model.add(Flatten())
  layer = tf.layers.flatten(layer)
  model.add(Dense(ENCODER_DIM))
  layer = tf.layers.dense(layer, ENCODER_DIM)
  model.add(Dense(4*4*1024))
  layer = tf.layers.dense(layer, 4*4*1024)
  model.add(Reshape((4,4,1024)))
  layer = tf.reshape(layer, [-1,4,4,1024])
  model.add(UpSampling2D((2,2)))
  layer = tf.image.resize_images(layer, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  def upsample(inputs, filters, out_size):
    layer = tf.layers.conv2d_transpose(inputs, filters, (5,5), padding='same')
    return tf.image.resize_images(layer, size=(out_size,out_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  pdb.set_trace()
  model.add(Conv2D(512, (5,5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  layer = upsample(layer, 512, 16)
  model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  layer = upsample(layer, 512, 32)
  model.add(Conv2D(128, (5,5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  layer = upsample(layer, 512, 64)
  model.add(Conv2D(3, (5,5), activation='sigmoid', padding='same'))
  layer = tf.layers.conv2d_transpose(layer, 3, (5,5), activation=tf.sigmoid, padding='same')

  return model, layer

def conv(model, filters):
  model.add(Conv2D(filters, kernel_size=5, strides=2, padding='same'))
  model.add(LeakyReLU(0.1))

def upscale(model, filters):
  # 4,4,1024
  model.add(Conv2D(filters * 4, kernel_size=3, padding='same'))
  # 4,4,2048
  model.add(LeakyReLU(0.1))
  # 8,8,512
  model.add(PixelShuffler())

model, layer = AutoEncoder()
model_a = model
model_b = model

#model.compile(optimizer='adadelta', loss='binary_crossentropy')
#model.compile(optimizer='adadelta', loss='mean_absolute_error')
optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
model_a.compile(optimizer=optimizer, loss='mean_absolute_error')
model_b.compile(optimizer=optimizer, loss='mean_absolute_error')

trump_images = load_images2("./photo/trump")
cage_images = load_images2("./photo/cage")

SAVE_FILE = 'models2/weights.h5f5'

import matplotlib.pyplot as plt
plt.ion() 
plt.show()

n = 4
plt.figure(figsize=(n,4))

def show_graph():
  decoded_images = model_a.predict(cage_images[:10])

  trump = [
    './photo/trump/1122709150.jpg',
    './photo/trump/1155971140.jpg',
    './photo/trump/1155971350.jpg',
    './photo/trump/1228920080.jpg']

  cage = [
    'photo/cage/102667226.jpg', 
    'photo/cage/102668242.jpg', 
    'photo/cage/102669741.jpg', 
    'photo/cage/102670060.jpg',
  ]

  def load_singles(model, filenames):
    raw = []
    decoded = []
    for fn in filenames:
      img = load_img(fn)
      img = img.resize((INPUT_DIM, INPUT_DIM))
      pix = np.array([np.array(img)])
      pix = pix / 255.0
      raw.append(pix[0])
      decoded.append(model.predict(pix)[0])
    return [raw, decoded]

  raw_cage, decoded_cage = load_singles(model_a, cage)
  raw_trump, decoded_trump = load_singles(model_a, trump)

  def no_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  for i in range(n):
    ax = plt.subplot(4,4,i+1)
    plt.imshow(decoded_images[i]) 
    no_axis(ax)

    ax = plt.subplot(4,4,4+i+1)
    plt.imshow(decoded_cage[i]) 
    no_axis(ax)

    ax = plt.subplot(4,4,8+i+1)
    plt.imshow(raw_trump[i]) 
    no_axis(ax)

    ax = plt.subplot(4,4,12+i+1)
    plt.imshow(decoded_trump[i]) 
    no_axis(ax)

  plt.pause(0.001)

do_saves = False

if do_saves and os.path.isfile(SAVE_FILE):
  model.load_weights(SAVE_FILE)

cp = ModelCheckpoint(SAVE_FILE)
class ShowSamples(Callback):
  def on_epoch_end(self, epoch, log={}):
    show_graph()

if do_saves:
    callbacks = [cp, ShowSamples()]
else:
    callbacks = [ShowSamples()]

model_a.fit(cage_images, cage_images,
  steps_per_epoch=2000 // BATCH_SIZE,
  epochs=1,
  #callbacks=callbacks,
  #callbacks=[cp],
  #validation_data=validation_generator,
  #validation_steps=800 // BATCH_SIZE
  )

model_b.fit(trump_images, temp_images,
  steps_per_epoch=2000 // BATCH_SIZE,
  epochs=1,
  callbacks=callbacks,
  #callbacks=[cp],
  #validation_data=validation_generator,
  #validation_steps=800 // BATCH_SIZE
  )

show_graph()
