import tensorflow as tf
from tensorflow import keras

import numpy as np
import random
import string
from PIL import Image, ImageFont, ImageDraw
import os

def random_string(slen):
    letters = string.ascii_letters + "    "
    return ''.join(random.choice(letters) for _ in range(slen))

IMAGE_SIZE = (256, 256)
FONT_START = 4
FONT_SIZE_RANGE = range(FONT_START, 43)
TARGET_FONT_SIZE = 16
DATA_DIR = './data'
BATCH_SIZE = 20

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


def get_image(fname, fno):
    return Image.open("{0}/{1}/{2}.png".format(DATA_DIR, fname, fno))

def preprocess_image(image):
    return image

def image_generator(files,label_file, batch_size = 64):
    
    while True:
        choices = random.sample(range(0,NUMBER_OF_FILES))
        batch_x = []
        batch_y = [] 
                                                            
        # Read in each input, perform preprocessing and get labels
        for choice in choices:
            x = get_image("x", choice)
            y = get_image("y", choice)
            x = preprocess_image(image=x)
            batch_x += [ x ]
            batch_y += [ y ]
            # Return a tuple of (input,output) to feed the network

        batch_x = np.array( batch_x )
        batch_y = np.array( batch_y )
            
        yield( batch_x, batch_y )

image = get_image("x", 1)
print(np.array(image).shape)

input_layer = keras.layers.Input(shape=IMAGE_SIZE+(3,))
model = model(
