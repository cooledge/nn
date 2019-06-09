import tensorflow as tf
from tensorflow import keras

import numpy as np
import random
import string
from PIL import Image, ImageFont, ImageDraw
import os
import sys

if "../" not in sys.path:
    sys.path.append("../lib")
from helpers import split_by_percentage

import pdb

def random_string(slen):
    letters = string.ascii_letters + "    "
    return ''.join(random.choice(letters) for _ in range(slen))

IMAGE_SIZE = (256, 256)
FONT_START = 4
FONT_SIZE_RANGE = range(FONT_START, 43)
TARGET_FONT_SIZE = 16
DATA_DIR = './data'
BATCH_SIZE = 16

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    os.makedirs("{0}/x".format(DATA_DIR))
    os.makedirs("{0}/y".format(DATA_DIR))

fonts = [ImageFont.truetype("LiberationSans-Regular.ttf", i) for i in FONT_SIZE_RANGE]
def get_font(size):
    return fonts[size-FONT_START]

font_y = get_font(TARGET_FONT_SIZE)

def save_image(fn, sample_counter, font_size):
    img_x = Image.new("RGB", IMAGE_SIZE, "white")
    draw_x = ImageDraw.Draw(img_x)
    draw_x.text((0, 0), text, "black", font=get_font(font_size))
    img_x.save("{0}/{1}/{2}.png".format(DATA_DIR, fn, sample_counter))

sample_counter = 0
for font_size in FONT_SIZE_RANGE:
    text = "Sample Text"
    save_image("x", sample_counter, font_size)
    save_image("y", sample_counter, TARGET_FONT_SIZE)
    sample_counter += 1

NUMBER_OF_FILES = len(os.listdir("{0}/x".format(DATA_DIR)))

def get_image(fname, fno):
    return Image.open("{0}/{1}/{2}.png".format(DATA_DIR, fname, fno))

def preprocess_image(image):
    return image

data = [i for i in range(0, NUMBER_OF_FILES)]
np.random.shuffle(data)
data_training, data_validate, data_test = split_by_percentage(data, [80, 10, 10])

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
            batch_x += [ np.array(x.getdata()).reshape(IMAGE_SIZE+(3,)) ]
            batch_y += [ np.array(y.getdata()).reshape(IMAGE_SIZE+(3,)) ]
            i += 1

        batch_x = np.array( batch_x )
        batch_y = np.array( batch_y )
            
        yield( batch_x, batch_y )

image = get_image("x", 1)
print(np.array(image).shape)

input_layer = keras.layers.Input(shape=IMAGE_SIZE+(3,))
model = keras.Model(input_layer, input_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_absolute_error')
pdb.set_trace()
model.fit_generator(image_generator(data_training), steps_per_epoch=int(len(data_training)/BATCH_SIZE))

