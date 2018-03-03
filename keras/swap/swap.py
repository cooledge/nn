import tensorflow as tf
import pdb
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam

INPUT_DIM = 64
INPUT_SIZE=(INPUT_DIM, INPUT_DIM, 3)
ENCODER_DIM = 1024
BATCH_SIZE = 16

def load_images(dir):
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1./255)
  train_generator = train_datagen.flow_from_directory(dir+"/train", target_size=(INPUT_DIM,INPUT_DIM), batch_size=BATCH_SIZE, class_mode='binary')
  validation_generator = train_datagen.flow_from_directory(dir+"/validation", target_size=(INPUT_DIM,INPUT_DIM), batch_size=BATCH_SIZE, class_mode='binary')
  return train_generator, validation_generator

def upscale(model, filters):
  model.add(Conv2D(filters * 4, kernel_size=3, padding='same'))
  model.add(LeakyReLU(0.1))

def AutoEncoder():
  model = Sequential()

  # Encoder

  model.add(Conv2D(128, kernel_size=5, strides=2, padding='same', input_shape=INPUT_SIZE))
  model.add(LeakyReLU(0.1))

  model.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
  model.add(LeakyReLU(0.1))

  model.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
  model.add(LeakyReLU(0.1))

  model.add(Conv2D(1024, kernel_size=5, strides=2, padding='same'))
  model.add(LeakyReLU(0.1))

  model.add(Flatten())
  model.add(Dense(ENCODER_DIM))
  model.add(Dense(4*4*ENCODER_DIM))
  model.add(Reshape((4,4,ENCODER_DIM)))
  upscale(model, 512)

  # Decoder
  model.add(Reshape((8,8,512)))
  upscale(model, 256)
  upscale(model, 128)
  upscale(model, 64)

  return model

model = AutoEncoder()

model.compile(optimizer='adadelta', loss='binary_crossentropy')

train_generator, validation_generator = load_images("photo/cage")

model.fit_generator(train_generator, 
  steps_per_epoch=2000 // BATCH_SIZE,
  epochs=50,
  validation_data=validation_generator,
  validation_steps=800 // BATCH_SIZE)
    
