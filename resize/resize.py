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
parser.add_argument("--epochs", type=int, default=1, help="epochs")
parser.add_argument("--retrain", action='store_true', default=False, help="retrain the nn")
parser.add_argument("--clean", action='store_true', default=False, help="regenerate data and weights")
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
        for font_size in FONT_SIZE_RANGE:
            save_image("x", sample_counter, font_size, text)
            save_image("y", sample_counter, TARGET_FONT_SIZE, text)
            sample_counter += 1

if args.clean:
    generate_data()

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

input_layer = keras.layers.Input(shape=IMAGE_SIZE+(3,))
output_layer = keras.layers.Dense(3)(input_layer)
model = keras.Model(input_layer, output_layer)

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
  
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_absolute_error')
#model.fit_generator(image_generator(data_training), steps_per_epoch=int(len(data_training)/BATCH_SIZE), validation_data=image_generator(data_validation), validation_steps=int(len(data_validation)/BATCH_SIZE))

predictions = model.predict_generator(image_generator(data_test), steps=int(len(data_test)/BATCH_SIZE))
pdb.set_trace()
pdb.set_trace()

for i in range(len(data_test)):
    x = get_image('x', i)
    y = predictions[i]
    show_data(x, y)


