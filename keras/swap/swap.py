import tensorflow as tf
import numpy as np
import pdb
import os.path
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

def AutoEncoder_FC():
  model = Sequential()

  model.add(Flatten(input_shape=INPUT_SIZE))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(np.prod(INPUT_SIZE), activation='sigmoid'))
  model.add(Reshape(INPUT_SIZE))

  return model

def AutoEncoder_FC():
  model = Sequential()

  # Encoder
  model.add(Flatten(input_shape=INPUT_SIZE))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))

  # Decoder
  model.add(Dense(64, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(np.prod(INPUT_SIZE), activation='sigmoid'))
  model.add(Reshape(INPUT_SIZE))

  return model

def AutoEncoder():
  model = Sequential()

  # Encoder
  model.add(Conv2D(128, (5,5), activation='relu', padding='same', input_shape=INPUT_SIZE))
  model.add(MaxPooling2D(2,2, padding='same'))
  model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
  model.add(MaxPooling2D(2,2, padding='same'))
  model.add(Conv2D(512, (5,5), activation='relu', padding='same'))
  model.add(MaxPooling2D(2,2, padding='same'))
 
  model.add(Flatten())
  pdb.set_trace()
  model.add(Dense(ENCODER_DIM))
  model.add(Dense(4*4*1024))
  model.add(Reshape((4,4,1024)))
  model.add(UpSampling2D((2,2)))

  # Decoder
  model.add(Conv2D(512, (5,5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Conv2D(128, (5,5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Conv2D(3, (5,5), activation='sigmoid', padding='same'))

  return model

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

def AutoEncoder1():
  model = Sequential()

  # Encoder
  model.add(Conv2D(128, (5,5), strides=2, padding='same', input_shape=INPUT_SIZE))
  model.add(LeakyReLU(0.1))
  conv(model, 256)
  conv(model, 512)
  conv(model, 1024)
  model.add(Flatten())
  model.add(Dense(ENCODER_DIM))
  model.add(Dense(4*4*1024))
  model.add(Reshape((4,4,1024)))
  upscale(model, 512)
  # should be 8,8,512

  # Decoder
  upscale(model, 256)
  upscale(model, 128)
  upscale(model, 64)
  model.add(Conv2D(3, kernel_size=5, padding='same', activation='sigmoid'))

  return model

model = AutoEncoder()

#model.compile(optimizer='adadelta', loss='binary_crossentropy')
#model.compile(optimizer='adadelta', loss='mean_absolute_error')
optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
model.compile(optimizer=optimizer, loss='mean_absolute_error')

train_generator, validation_generator = load_images("./photo")

SAVE_FILE = 'models2/weights.h5f5'

import matplotlib.pyplot as plt
plt.ion() 
plt.show()

n = 4
plt.figure(figsize=(n,4))

def show_graph():
  decoded_images = model.predict_generator(validation_generator)

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

  def load_singles(filenames):
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

  raw_cage, decoded_cage = load_singles(cage)
  raw_trump, decoded_trump = load_singles(trump)

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

model.fit_generator(train_generator, 
  steps_per_epoch=2000 // BATCH_SIZE,
  epochs=100000,
  callbacks=callbacks,
  #callbacks=[cp],
  validation_data=validation_generator,
  validation_steps=800 // BATCH_SIZE)

