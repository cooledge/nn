import tensorflow as tf
from tensorflow import keras

import numpy as np
import random
import string
from PIL import Image, ImageFont, ImageDraw
import os
import sys
import argparse

if "../" not in sys.path:
  sys.path.append("../lib")
from helpers import split_by_percentage

import pdb

parser = argparse.ArgumentParser(description="Text Resizer")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--retrain", action='store_true', default=False, help="retrain the nn")
parser.add_argument("--clean", action='store_true', default=False, help="regenerate data and weights")
parser.add_argument("--model", type=int, default=1, help="Model number to run with")
parser.add_argument("--show", action='store_true', default=False, help="show the images test and predictions")
args = parser.parse_args()

def random_string(slen):
  letters = string.ascii_letters + "    "
  return ''.join(random.choice(letters) for _ in range(slen))

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 32
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
NUMBER_OF_COLORS = 1
FONT_START = 4
FONT_END = 4
FONT_SIZE_RANGE = range(FONT_START, FONT_END+1)
TARGET_FONT_SIZE = 24
GENERATE_FONT_SIZE_RANGE = range(FONT_START, max(FONT_END, TARGET_FONT_SIZE)+1)
DATA_DIR = './data'
BATCH_SIZE = args.batch_size
N_STRINGS = 100
ENCODING_DIM = 1024
WEIGHTS_FILE = "weights_{0}".format(args.model)

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
  os.makedirs("{0}/x".format(DATA_DIR))
  os.makedirs("{0}/y".format(DATA_DIR))
  args.clean = True

fonts = [ImageFont.truetype("LiberationSans-Regular.ttf", i) for i in GENERATE_FONT_SIZE_RANGE]
def get_font(size):
  return fonts[size-FONT_START]

font_y = get_font(TARGET_FONT_SIZE)

def mean_absolute_error(imageA, imageB):
  mae = np.sum(np.absolute(imageB.astype("float") - imageA.astype("float")))
  mae /= float(imageA.shape[0] * imageA.shape[1] * 255) 
  return mae

def data_to_image(data):
  data *= 255
  if NUMBER_OF_COLORS == 1:
    data.resize(IMAGE_HEIGHT, IMAGE_WIDTH)
  else:
    data.resize(IMAGE_HEIGHT, IMAGE_WIDTH, NUMBER_OF_COLORS)
  return Image.fromarray(np.uint8(data))

def save_image(fn, sample_counter, font_size, text):
  img_x = Image.new("RGB", IMAGE_SIZE, "white")
  draw_x = ImageDraw.Draw(img_x)
  draw_x.text((0, 0), text, "black", font=get_font(font_size))
  img_x.save("{0}/{1}/{2}.png".format(DATA_DIR, fn, sample_counter))

def generate_data():
  sample_counter = 0
  for _ in range(N_STRINGS):
      text = random_string(10)
      print(text)
      for font_size in FONT_SIZE_RANGE:
          save_image("x", sample_counter, font_size, text)
          save_image("y", sample_counter, TARGET_FONT_SIZE, text)
          sample_counter += 1

def safe_remove(fn):
  import shutil
  try:
    shutil.rmtree(fn)
  except FileNotFoundError:
    0 

if args.retrain:
  safe_remove(WEIGHTS_FILE)

if args.clean:
  generate_data()
  safe_remove(WEIGHTS_FILE)
  
DIR_X = "{0}/x".format(DATA_DIR)
NUMBER_OF_FILES = len([f for f in os.listdir(DIR_X) if os.path.isfile(os.path.join(DIR_X, f))])

def get_image(fname, fno):
  image = Image.open("{0}/{1}/{2}.png".format(DATA_DIR, fname, fno))
  image = image.convert('1')
  image = np.array(image.getdata()).reshape(IMAGE_SIZE+(1,))
  image = image / 255.
  return image

def preprocess_image(image):
  return image

data = [i for i in range(0, NUMBER_OF_FILES)]
np.random.shuffle(data)
data_training, data_validation, data_test = split_by_percentage(data, [34, 33, 33])

def image_generator(data):
  
  while True: 
      i = 0 
      while i+BATCH_SIZE < len(data):
          batch_x = []
          batch_y = [] 
                                                              
          for _ in range(BATCH_SIZE):
              choice = data[i]
              x = get_image("x", choice)
              y = get_image("y", choice)
              x = preprocess_image(image=x)
              batch_x += [ x ]
              batch_y += [ y ]
              i += 1

          batch_x = np.array( batch_x )

          batch_y = np.array( batch_y )
              
          yield( batch_x, batch_y )

#image = get_image("x", 1)
#print(np.array(image).shape)

def super_model():
  # 256,32,3
  input_layer = keras.layers.Input(shape=IMAGE_SIZE+(NUMBER_OF_COLORS,))
  # 256, 32, 128
  output_layer = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(input_layer)
  # 128, 16, 128
  output_layer = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(output_layer)
  # 256, 16, 128
  output_layer = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(output_layer)
  # 64, 8, 128
  output_layer = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(output_layer)

  # 
  output_layer = keras.layers.Flatten()(output_layer)
  # 1024
  output_layer = keras.layers.Dense(ENCODING_DIM)(output_layer)

  # 16384
  output_layer = keras.layers.Dense(4*4*1024)(output_layer)
  # 4,4,1024
  output_layer = keras.layers.Reshape((4,4,1024))(output_layer)
  # 8,4,1024
  output_layer = keras.layers.UpSampling2D((2,1))(output_layer)
  # 8,4,512
  output_layer = keras.layers.Conv2D(512, (3,3), activation='relu', padding='same')(output_layer)
  # 16,4,512
  output_layer = keras.layers.UpSampling2D((2,1))(output_layer)
  # 16,4,256
  output_layer = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(output_layer)
  # 64,16,,256
  output_layer = keras.layers.UpSampling2D((4,4))(output_layer)
  # 64,16,3
  output_layer = keras.layers.Conv2D(NUMBER_OF_COLORS, (3,3), activation='relu', padding='same')(output_layer)
  # 256,32,3
  output_layer = keras.layers.UpSampling2D((4,2))(output_layer)

  return keras.Model(input_layer, output_layer)

def simple_model():
  # 256,256,3
  il = keras.layers.Input(shape=IMAGE_SIZE+(NUMBER_OF_COLORS,))
  # 256, 256, 128
  ol = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(il)

  ol = keras.layers.Flatten()(ol)
  # 1024
  ol = keras.layers.Dense(ENCODING_DIM)(ol)

  # 16384
  ol = keras.layers.Dense(256*IMAGE_HEIGHT*1024)(ol)
  # 256,256,1024
  ol = keras.layers.Reshape((256,IMAGE_HEIGHT,1024))(ol)
  # 256,256,3
  ol = keras.layers.Conv2D(NUMBER_OF_COLORS, (3,3), activation='relu', padding='same')(ol)

  return keras.Model(il, ol)

def split_model():
  # 256,256,3
  il = keras.layers.Input(shape=IMAGE_SIZE+(NUMBER_OF_COLORS,))
  # 256, 256, 128
  ol1 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(il)
  ol2 = keras.layers.Conv2D(128, (5,5), activation='relu', padding='same')(il)

  # 256, 256, 128+128
  ol = keras.layers.Concatenate()([ol1, ol2])

  # 128, 128, 256
  ol = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(ol)

  ol1 = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(ol)
  ol2 = keras.layers.Conv2D(32, (5,5), activation='relu', padding='same')(ol)

  # 128, 128, 64
  ol = keras.layers.Concatenate()([ol1, ol2])

  # 64, 64, 64
  ol = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(ol)

  ol = keras.layers.Flatten()(ol)
  # 1024
  ol = keras.layers.Dense(ENCODING_DIM)(ol)

  # 
  ol = keras.layers.Dense(32*32*128)(ol)
  # 32, 32, 128
  ol = keras.layers.Reshape((32,32,128))(ol)
  # 64, 64, 128

  ol = keras.layers.UpSampling2D((2,2))(ol)
  # 64, 64, 64
  ol = keras.layers.Conv2D(64, (2,2), activation='relu', padding='same')(ol)

  # 128, 128, 64
  ol = keras.layers.UpSampling2D((2,2))(ol)
  # 128, 128, 16
  ol = keras.layers.Conv2D(16, (2,2), activation='relu', padding='same')(ol)

  # 256, 256, 16
  ol = keras.layers.UpSampling2D((2,2))(ol)
   
  # 256, 256, 3 
  ol = keras.layers.Conv2D(NUMBER_OF_COLORS, (2,2), activation='relu', padding='same')(ol)

  return keras.Model(il, ol)

def split_model_v2():
  # 256,256,3
  il = keras.layers.Input(shape=IMAGE_SIZE+(NUMBER_OF_COLORS,))
  # 256, 256, 128
  ol1 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(il)
  ol2 = keras.layers.Conv2D(128, (5,5), activation='relu', padding='same')(il)

  # 256, 256, 128+128
  ol = keras.layers.Concatenate()([ol1, ol2])

  # 128, 128, 256
  ol = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(ol)

  ol1 = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(ol)
  ol2 = keras.layers.Conv2D(32, (5,5), activation='relu', padding='same')(ol)

  # 128, 128, 64
  ol = keras.layers.Concatenate()([ol1, ol2])

  # 64, 64, 64
  ol = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(ol)

  ol = keras.layers.Flatten()(ol)
  # 1024
  ol = keras.layers.Dense(ENCODING_DIM)(ol)

  # 
  ol = keras.layers.Dense(32*32*128)(ol)
  # 32, 32, 128
  ol = keras.layers.Reshape((32,32,128))(ol)

  # 64, 64, 128
  ol = keras.layers.UpSampling2D((2,2))(ol)
  ol1 = keras.layers.Conv2D(64, (2,2), activation='relu', padding='same')(ol)
  ol2 = keras.layers.Conv2D(64, (5,5), activation='relu', padding='same')(ol)
  # 64, 64, 128
  ol = keras.layers.Concatenate()([ol1, ol2])

  # 128, 128, 128
  ol = keras.layers.UpSampling2D((2,2))(ol)
  
  ol1 = keras.layers.Conv2D(16, (2,2), activation='relu', padding='same')(ol)
  ol2 = keras.layers.Conv2D(16, (5,5), activation='relu', padding='same')(ol)
  # 128, 128, 32
  ol = keras.layers.Concatenate()([ol1, ol2])
  # 256, 256, 32
  ol = keras.layers.UpSampling2D((2,2))(ol)
  ol = keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(ol)
   
  # 256, 256, 3 
  ol = keras.layers.Conv2D(NUMBER_OF_COLORS, (2,2), activation='relu', padding='same')(ol)

  return keras.Model(il, ol)

def null_model():
  input_layer = keras.layers.Input(shape=IMAGE_SIZE+(NUMBER_OF_COLORS,))
  output_layer = keras.layers.Dense(NUMBER_OF_COLORS)(input_layer)
  return keras.Model(input_layer, output_layer)

'''
print(null_model().output_shape)
print(simple_model().output_shape)
print(super_model().output_shape)
'''
if args.model == 1:
  model = null_model()
if args.model == 2:
  model = simple_model()
if args.model == 3:
  model = super_model()
if args.model == 4:
  model = split_model()
if args.model == 5:
  model = split_model_v2()

model.summary()

if args.show:
  import matplotlib.pyplot as plt
  plt.ion()
  plt.show()

  n_rows = 3
  n_cols = 1
  plt.figure(figsize=(n_rows,n_cols))
  n_images = 1

  def no_axis(ax):
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

  def show_data(x, y, p, pause_time=1.0):
      ax = plt.subplot(n_rows,n_cols,1)
      plt.imshow(data_to_image(x))
      no_axis(ax)

      ax = plt.subplot(n_rows,n_cols,2)
      plt.imshow(data_to_image(y))
      no_axis(ax)

      ax = plt.subplot(n_rows,n_cols,3)
      plt.imshow(data_to_image(p))
      no_axis(ax)

      #mae = mean_absolute_error(y, y) 
      #print(mae)

      plt.pause(1)

'''
pdb.set_trace()
show_data(get_image("x", 0), get_image("y", 0))
pdb.set_trace()
show_data(get_image("y", 0), get_image("x", 0))
pdb.set_trace()
'''

training_steps_per_epoch = int(len(data_training)/BATCH_SIZE)
validation_steps_per_epoch = int(len(data_validation)/BATCH_SIZE)
if training_steps_per_epoch == 0:
  print("Not enough training data")
  exit(-1)

if validation_steps_per_epoch == 0:
  print("Not enough validation data")
  exit(-1)

if os.path.exists(WEIGHTS_FILE):
  model.load_weights(WEIGHTS_FILE)
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_absolute_error')
else:
  use_tape = False
  if use_tape:
    optimizer=tf.keras.optimizers.Adam()
    generator = image_generator(data_training)
    for i in range(training_steps_per_epoch):
      compute_apply_gradients(model, next(generator) , optimizer)
  else:
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
    model.fit_generator(image_generator(data_training), epochs=args.epochs, steps_per_epoch=training_steps_per_epoch, validation_data=image_generator(data_validation), validation_steps=validation_steps_per_epoch)
  
  model.save(WEIGHTS_FILE)

predictions = model.predict_generator(image_generator(data_test), steps=int(len(data_test)/BATCH_SIZE))

if args.show:
  for i in range(len(data_test)):
    x = get_image('x', i)
    y = get_image('y', i)
    p = predictions[i]
    show_data(x, y, p)

