import tensorflow as tf
import numpy as np
import pdb
import sys
import os.path
import random
from PIL import Image
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape, MaxPooling2D, Conv2D, UpSampling2D, Lambda
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

def Encoder():
  model = Sequential()

  model.add(Conv2D(128, (5,5), activation='relu', padding='same', input_shape=INPUT_SIZE))
  model.add(MaxPooling2D(2,2, padding='same'))
  model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
  model.add(MaxPooling2D(2,2, padding='same'))
  model.add(Conv2D(512, (5,5), activation='relu', padding='same'))
  model.add(MaxPooling2D(2,2, padding='same'))
 
  model.add(Flatten())
  model.add(Dense(ENCODER_DIM))
  
  return model

def Decoder(model):
  model.add(Dense(4*4*1024))
  model.add(Reshape((4,4,1024)))
  model.add(UpSampling2D((2,2)))

  model.add(Conv2D(512, (5,5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Conv2D(128, (5,5), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Conv2D(3, (5,5), activation='sigmoid', padding='same'))

  return model

def Categorizer(model):
  model.add(Dense(2))

def AutoEncoder():
  encoder = Encoder()
  encoder_copy = copy_model(encoder)
  autoencoder_A = Decoder(encoder)
  autoencoder_B = Decoder(encoder_copy)
  categorizer = Categorizer(encoder)
  return autoencoder_A, autoencoder_B, categorizer

def conv(filters):
  def block(x):
    x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    return x
  return block

def upscale(filters):
  def block(x):
    x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = PixelShuffler()(x)
    return x
  return block

model_a, model_b, model_cat = AutoEncoder()

#model.compile(optimizer='adadelta', loss='binary_crossentropy')
#model.compile(optimizer='adadelta', loss='mean_absolute_error')
optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
model_a.compile(optimizer=optimizer, loss='mean_absolute_error')
model_b.compile(optimizer=optimizer, loss='mean_absolute_error')

trump_images = load_images2("./photo/joe_video")
cage_images = load_images2("./photo/joey_video")

SAVE_FILE = 'models2/weights.h5f5'

import matplotlib.pyplot as plt
plt.ion() 
plt.show()

n_rows = 6
n_cols = 4
plt.figure(figsize=(n_rows,n_cols))
n_images = 4

random.shuffle(cage_images)
random.shuffle(trump_images)

def show_graph():
  trump = trump_images[0:4]
  cage = cage_images[0:4]

  decoded_cage = model_a.predict(cage)
  decoded_trump = model_b.predict(trump)
  decoded_trump_as_cage = model_a.predict(trump)
  decoded_cage_as_trump = model_b.predict(cage)

  def no_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  for i in range(n_images):
    ax = plt.subplot(n_rows,n_cols,i+1)
    plt.imshow(cage[i]) 
    no_axis(ax)

    ax = plt.subplot(n_rows,n_cols,4+i+1)
    plt.imshow(decoded_cage[i]) 
    no_axis(ax)

    ax = plt.subplot(n_rows,n_cols,8+i+1)
    plt.imshow(decoded_cage_as_trump[i]) 
    no_axis(ax)

    ax = plt.subplot(n_rows,n_cols,12+i+1)
    plt.imshow(trump[i]) 
    no_axis(ax)

    ax = plt.subplot(n_rows,n_cols,16+i+1)
    plt.imshow(decoded_trump[i]) 
    no_axis(ax)

    ax = plt.subplot(n_rows,n_cols,20+i+1)
    plt.imshow(decoded_trump_as_cage[i]) 
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

epochs = 100000

for epoch in range(epochs): 
  print('Epoch {0}'.format(epoch))

  random.shuffle(cage_images)
  random.shuffle(trump_images)
  n_batches = min(len(cage_images), len(trump_images)) // BATCH_SIZE

  n_steps = 1000 // BATCH_SIZE

  for step in range(n_steps):
    for batch_no in range(n_batches):
      batch_start = batch_no * BATCH_SIZE

      def train_one_step(images, model):
        batch = images[batch_start:batch_start+BATCH_SIZE]
        return model.train_on_batch(batch, batch)

      if batch_no % 2 == 0:
        loss_a = train_one_step(cage_images, model_a)
        loss_b = train_one_step(trump_images, model_b)
      else:
        loss_b = train_one_step(trump_images, model_b)
        loss_a = train_one_step(cage_images, model_a)

      sys.stdout.write("step: {0} loss_a: {1} loss_b: {2}\r".format(step, loss_a, loss_b))
      sys.stdout.flush()
      show_graph()

  print("\n")
   
# split file into images
# joins images back into file    
