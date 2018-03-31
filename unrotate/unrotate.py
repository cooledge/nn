import tensorflow as tf
import numpy as np
import cv2
import time
import math
import pdb
import os.path
import random
from umeyama import umeyama
from PIL import Image

INPUT_DIM = 64
INPUT_SIZE=(INPUT_DIM, INPUT_DIM, 3)
ENCODER_DIM = 1024
BATCH_SIZE = 32
SAVE_DIR = 'models'
SAVE_FILE = 'models/swap'

def si(image):
  cv2.imshow("", image)
  cv2.waitKey(1000)

def warp_images(images):
  warped_images = []
  target_images = []
  rotations = []
  for image in images:
    wi, ti, rotate = warp_image(image)
    warped_images.append(wi)
    target_images.append(ti)
    rotations.append(rotate)

  return warped_images, target_images, rotations

def warp_image(image, coverage=160):
  image = cv2.resize(image, (256,256))
  transformed_image, rotation, flip = random_transform( image )
  image, _, _ = random_transform( image, rotation=0, flip=flip )
  warped_img, target_img, img = random_warp( transformed_image, image, coverage )
  return warped_img, img, rotation

def random_transform(image, rotation_range=90, rotation=None, zoom_range=0.00, shift_range=0.00, random_flip=0.5, flip=None):
  h, w = image.shape[0:2]
  if rotation is None:
    rotation = np.random.uniform(-rotation_range, rotation_range)
  if rotation < 0:
    rotation = 360 + rotation
  scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
  tx = np.random.uniform(-shift_range, shift_range) * w
  ty = np.random.uniform(-shift_range, shift_range) * h
  mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
  mat[:, 2] += (tx, ty)
  result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)

  if flip or np.random.random() < random_flip:
    result = result[:, ::-1]
    flip = True
  return result, rotation, flip

# get pair of random warped images from aligned face image
def random_warp(transformed_image, image, coverage=160):
  assert transformed_image.shape == (256, 256, 3)
  assert image.shape == (256, 256, 3)
  range_ = np.linspace(128 - coverage//2, 128 + coverage//2, 5)
  mapx = np.broadcast_to(range_, (5, 5))
  mapy = mapx.T

  mapx = mapx + np.random.normal(size=(5, 5), scale=5)
  mapy = mapy + np.random.normal(size=(5, 5), scale=5)

  interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
  interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

  warped_image = cv2.remap(transformed_image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

  src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
  dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
  mat = umeyama(src_points, dst_points, True)[0:2]

  target_image = cv2.warpAffine(transformed_image, mat, (64, 64))
  image = cv2.warpAffine(image, mat, (64, 64))

  return warped_image, target_image, image

def load_image(filepath):
  try:
    image = Image.open(filepath).resize((INPUT_DIM, INPUT_DIM))
    image = np.array(image)
    return image / 255.
  except IOError:
    return None
  
def load_images(dir):
  files = os.listdir(dir)
  images = []
  for file in files:
    image = load_image(dir+"/"+file)
    if image is not None:
      np.array(images.append(image))
  return np.array(images)

def copy_model(model):
  copy = Sequential()
  for layer in model.layers:
    copy.add(layer)
  return copy

def tf_add_conv(inputs, filters):
  layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
  layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  return layer

def Encoder(layer):
  # Encoder
  layer = tf_add_conv(layer, 128)
  layer = tf_add_conv(layer, 256)
  layer = tf_add_conv(layer, 512)
 
  layer = tf.layers.flatten(layer)
  layer = tf.layers.dense(layer, ENCODER_DIM)
  return layer

def Decoder(layer):
  layer = tf.layers.dense(layer, 4*4*1024)
  layer = tf.reshape(layer, [-1,4,4,1024])
  layer = tf.image.resize_images(layer, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  def upsample(inputs, filters, out_size):
    layer = tf.layers.conv2d_transpose(inputs, filters, (5,5), padding='same', activation=tf.nn.relu)
    return tf.image.resize_images(layer, size=(out_size,out_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  layer = upsample(layer, 512, 16)
  layer = upsample(layer, 512, 32)
  layer = upsample(layer, 512, 64)
  layer = tf.layers.conv2d_transpose(layer, 3, (5,5), activation=tf.sigmoid, padding='same')

  return layer

def rotate_resolution():
  return 5

def number_of_rotates():
  return math.ceil(360 / rotate_resolution())

def rotate_to_one_hot_index(degrees):
  rr = rotate_resolution()
  return round(degrees/rr)

def one_hot_index_to_rotate(index):
  return index * rotate_resolution()

def rotate_to_one_hot(degrees):
  index = rotate_to_one_hot_index(degrees)
  return [1 if i == index else 0 for i in range(number_of_rotates())]  

def one_hot_to_rotate(one_hot):
  return one_hot_index_to_rotate(np.argmax(one_hot))

def Categorizer(layer):
  return tf.layers.dense(layer, number_of_rotates())

def AutoEncoder():
  model_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="signal_input")
  model_output = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="signal_output")
  encoder = Encoder(model_input)
  decoder = Decoder(encoder)
  categorizer = Categorizer(encoder)
  return decoder, categorizer, model_input, model_output

model_autoencoder, model_categorizer, model_input, model_output = AutoEncoder()
trump_images = load_images("./photo/trump")
cage_images = load_images("./photo/cage")

import matplotlib
import matplotlib.pyplot as plt
plt.ion() 
plt.show()

n_rows = 6
n_cols = 4
plt.figure(figsize=(n_rows,n_cols))
n_images = 4

def show_graph(sess):
  def predict(images, autoencoder):
    placeholders ={ model_input: images }
    return sess.run(autoencoder, placeholders)

  def predict_cat(images, categorizer):
    placeholders = { model_input: images }
    one_hots = sess.run(model_prob_cat, placeholders)
    return [one_hot_to_rotate(one_hot) for one_hot in one_hots]

  test_images = inputs[0:4]
  test_rotations = rotations[0:4]
  decoded_cage = predict(test_images[0:4], model_autoencoder)
  decoded_angles = predict_cat(test_images[0:4], model_categorizer)

  def no_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  def plot_image(col, image):
    ax = plt.subplot(n_rows,n_cols,(4*col)+i+1)
    plt.imshow(image) 
    no_axis(ax)

  def plot_rotate(col, rotate):
    ax = plt.subplot(n_rows,n_cols,(4*col)+i+1)
    ax.text(0, 0, str(round(rotate))+"    ", fontsize=15, backgroundcolor='w')
    no_axis(ax)

  for i in range(n_images):
    plot_image(0, test_images[i])
    plot_image(1, decoded_cage[i])
    plot_rotate(2, test_rotations[i])
    plot_rotate(3, decoded_angles[i])

  plt.pause(0.001)

saver = tf.train.Saver()

model_prob_cat = tf.nn.softmax(model_categorizer)
model_output_cat = tf.placeholder(tf.float32, shape=[None, number_of_rotates()], name="model_output_cat")
model_loss_cat = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_output_cat, logits=model_categorizer))
model_optimizer_cat = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op_cat = model_optimizer_cat.minimize(model_loss_cat)

model_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(tf.layers.flatten(model_output), tf.layers.flatten(model_autoencoder)))
model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op = model_optimizer.minimize(model_loss)

images = cage_images
indexes = [i for i in range(len(images))]

def get_batch(indexes, start, end, x):
  value = [x[indexes[i]] for i in range(start, end)]
  return value

epochs = 10000
steps = 50
batches = len(images) // BATCH_SIZE

sess = tf.Session()

sess.run(tf.global_variables_initializer())
saved_model_path = tf.train.latest_checkpoint(SAVE_DIR)
if False and saved_model_path:
  saver.restore(sess, saved_model_path)

#show_graph(sess)

last_time = time.time()
for epoch in range(epochs):
  print("\nEpoch {0} seconds {1}".format(epoch, time.time()-last_time))
  last_time = time.time()
  random.shuffle(indexes)
  timages = [random_transform(image) for image in images]
  for step in range(steps):
    for batch in range(batches):
      start = batch*BATCH_SIZE
      end = (batch+1)*BATCH_SIZE

      batch = get_batch(indexes, start, end, images)
      inputs, outputs, rotations = warp_images(batch)
      one_hots = [rotate_to_one_hot(degrees) for degrees in rotations]
      placeholders = {
        model_input: inputs,
        model_output: outputs,
        model_output_cat: one_hots
      }
      # model_loss_cat, model_train_op_cat
      loss, loss_cat, _, _ = sess.run([model_loss, model_loss_cat, model_train_op, model_train_op_cat], placeholders)

    print("Step: {0} loss_a: {1}/{2}\r".format(step, loss, loss_cat))
    show_graph(sess)
    saver.save(sess, SAVE_FILE)
