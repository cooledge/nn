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
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--retrain", action='store_true', default=False, help="retrain the nn")
parser.add_argument("--clean", action='store_true', default=False, help="regenerate data and weights")
parser.add_argument("--model", type=int, default=1, help="Model number to run with")
parser.add_argument("--show", action='store_true', default=False, help="show the images test and predictions")
args = parser.parse_args()

def random_string(slen):
    letters = string.ascii_letters + "    "
    return ''.join(random.choice(letters) for _ in range(slen))

IMAGE_SIZE = (256, 256)
FONT_START = 4
FONT_SIZE_RANGE = range(FONT_START, 43)
TARGET_FONT_SIZE = 16
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

fonts = [ImageFont.truetype("LiberationSans-Regular.ttf", i) for i in FONT_SIZE_RANGE]
def get_font(size):
    return fonts[size-FONT_START]

font_y = get_font(TARGET_FONT_SIZE)

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
   try:
       os.remove(fn)
   except FileNotFoundError:
        0

if args.retrain:
    safe_remove(WEIGHTS_FILE)

if args.clean:
    generate_data()
    safe_remove(WEIGHTS_FILE)
    

NUMBER_OF_FILES = len(os.listdir("{0}/x".format(DATA_DIR)))

def get_image(fname, fno):
    return np.array(Image.open("{0}/{1}/{2}.png".format(DATA_DIR, fname, fno)).getdata()).reshape(IMAGE_SIZE+(3,))

def preprocess_image(image):
    return image

data = [i for i in range(0, NUMBER_OF_FILES)]
np.random.shuffle(data)
data_training, data_validation, data_test = split_by_percentage(data, [80, 10, 10])

def image_generator(data):
   
    i = 0 
    while True:
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

image = get_image("x", 1)
print(np.array(image).shape)

def super_model():
    # 256,256,3
    input_layer = keras.layers.Input(shape=IMAGE_SIZE+(3,))
    # 256, 256, 128
    output_layer = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(input_layer)
    # 128, 128, 128
    output_layer = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(output_layer)
    # 256, 256, 128
    output_layer = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(output_layer)
    # 64, 64, 128
    output_layer = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(output_layer)

    # 
    output_layer = keras.layers.Flatten()(output_layer)
    # 1024
    output_layer = keras.layers.Dense(ENCODING_DIM)(output_layer)

    # 16384
    output_layer = keras.layers.Dense(4*4*1024)(output_layer)
    # 4,4,1024
    output_layer = keras.layers.Reshape((4,4,1024))(output_layer)
    # 8,8,1024
    output_layer = keras.layers.UpSampling2D((2,2))(output_layer)
    # 8,8,512
    output_layer = keras.layers.Conv2D(512, (3,3), activation='relu', padding='same')(output_layer)
    # 16,16,512
    output_layer = keras.layers.UpSampling2D((2,2))(output_layer)
    # 8,8,256
    output_layer = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(output_layer)
    # 64,64,256
    output_layer = keras.layers.UpSampling2D((4,4))(output_layer)
    # 64,64,3
    output_layer = keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(output_layer)
    # 256,256,3
    output_layer = keras.layers.UpSampling2D((4,4))(output_layer)

    return keras.Model(input_layer, output_layer)

def simple_model():
    # 256,256,3
    il = keras.layers.Input(shape=IMAGE_SIZE+(3,))
    # 256, 256, 128
    ol = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(il)

    ol = keras.layers.Flatten()(ol)
    # 1024
    ol = keras.layers.Dense(ENCODING_DIM)(ol)

    # 16384
    ol = keras.layers.Dense(256*256*1024)(ol)
    # 256,256,1024
    ol = keras.layers.Reshape((256,256,1024))(ol)
    # 256,256,3
    ol = keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(ol)

    return keras.Model(il, ol)

def split_model():
    # 256,256,3
    il = keras.layers.Input(shape=IMAGE_SIZE+(3,))
    # 256, 256, 128
    ol1 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(il)
    ol2 = keras.layers.Conv2D(128, (5,5), activation='relu', padding='same')(il)

    ol = keras.layers.Concatenate([ol1, ol2])(il)

    ol = keras.layers.Flatten()(ol)
    # 1024
    ol = keras.layers.Dense(ENCODING_DIM)(ol)

    # 16384
    ol = keras.layers.Dense(256*256*1024)(ol)
    # 256,256,1024
    ol = keras.layers.Reshape((256,256,1024))(ol)
    # 256,256,3
    ol = keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(ol)

    return keras.Model(il, ol)

def null_model():
    input_layer = keras.layers.Input(shape=IMAGE_SIZE+(3,))
    output_layer = keras.layers.Dense(3)(input_layer)
    return keras.Model(input_layer, output_layer)

'''
print(null_model().output_shape)
print(simple_model().output_shape)
print(super_model().output_shape)
print(split_model().output_shape)
pdb.set_trace()
'''

if args.model == 1:
    model = null_model()
if args.model == 2:
    model = simple_model()
if args.model == 3:
    model = super_model()

if args.show:
    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()

    n_rows = 1
    n_cols = 2
    plt.figure(figsize=(n_rows,n_cols))
    n_images = 1

    def no_axis(ax):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def show_data(x, y, pause_time=1.0):
        ax = plt.subplot(n_rows,n_cols,1)
        plt.imshow(x)
        no_axis(ax)

        ax = plt.subplot(n_rows,n_cols,2)
        plt.imshow(y)
        no_axis(ax)

        plt.pause(1)

'''
pdb.set_trace()
show_data(get_image("x", 0), get_image("y", 0))
pdb.set_trace()
show_data(get_image("y", 0), get_image("x", 0))
pdb.set_trace()
'''
try:
    model.load_weights(WEIGHTS_FILE)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_absolute_error')
except:
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_absolute_error')
    model.fit_generator(image_generator(data_training), steps_per_epoch=int(len(data_training)/BATCH_SIZE), validation_data=image_generator(data_validation), validation_steps=int(len(data_validation)/BATCH_SIZE))
    model.save(WEIGHTS_FILE)

predictions = model.predict_generator(image_generator(data_test), steps=int(len(data_test)/BATCH_SIZE))

if args.show:
    for i in range(len(data_test)):
        x = get_image('x', i)
        y = predictions[i]
        show_data(x, y)


