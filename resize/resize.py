#import tensorflow as tf
#import tf.keras as keras

import random
import string
from PIL import Image, ImageFont, ImageDraw
import os

def random_string(slen):
    letters = string.ascii_letters + "    "
    return ''.join(random.choice(letters) for _ in range(slen))

FONT_START = 4
FONT_SIZE_RANGE = range(FONT_START, 43)
TARGET_FONT_SIZE = 16
DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

fonts = [ImageFont.truetype("LiberationSans-Regular.ttf", i) for i in FONT_SIZE_RANGE]
def get_font(size):
    return fonts[size-FONT_START]

font_y = get_font(TARGET_FONT_SIZE)

def save_image(fn, sample_counter, font_size):
    img_x = Image.new("RGB", (256, 256), "white")
    draw_x = ImageDraw.Draw(img_x)
    draw_x.text((0, 0), text, "black", font=get_font(font_size))
    img_x.save("{0}/{1}_{2}.png".format(DATA_DIR, fn, sample_counter))

sample_counter = 0
for font_size in FONT_SIZE_RANGE:
    text = "Sample Text"
    save_image("x", sample_counter, font_size)
    save_image("y", sample_counter, TARGET_FONT_SIZE)
    sample_counter += 1
