import tensorflow as tf
import numpy as np
import cv2
import time
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

def warp_images(images):
  warped_images = []
  target_images = []
  for image in images:
    wi, ti = warp_image(image)
    warped_images.append(wi)
    target_images.append(ti)

  return warped_images, target_images

def warp_image(image, coverage=160):
  image = cv2.resize(image, (256,256))
  image = random_transform( image )
  warped_img, target_img = random_warp( image, coverage )
  return warped_img, target_img

def random_transform(image, rotation_range=10, zoom_range=0.05, shift_range=0.05, random_flip=0.4):
  h, w = image.shape[0:2]
  rotation = np.random.uniform(-rotation_range, rotation_range)
  scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
  tx = np.random.uniform(-shift_range, shift_range) * w
  ty = np.random.uniform(-shift_range, shift_range) * h
  mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
  mat[:, 2] += (tx, ty)
  result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
  if np.random.random() < random_flip:
    result = result[:, ::-1]
  return result

# get pair of random warped images from aligned face image
def random_warp(image, coverage=160):
  assert image.shape == (256, 256, 3)
  range_ = np.linspace(128 - coverage//2, 128 + coverage//2, 5)
  mapx = np.broadcast_to(range_, (5, 5))
  mapy = mapx.T

  mapx = mapx + np.random.normal(size=(5, 5), scale=5)
  mapy = mapy + np.random.normal(size=(5, 5), scale=5)

  interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
  interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

  warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

  src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
  dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
  mat = umeyama(src_points, dst_points, True)[0:2]

  target_image = cv2.warpAffine(image, mat, (64, 64))
  return warped_image, target_image

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

def Decoder(layer, version, reset_layers):
  layer = tf.layers.dense(layer, 4*4*1024)
  layer = tf.reshape(layer, [-1,4,4,1024])
  layer = tf.image.resize_images(layer, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  gr = tf.get_default_graph()

  def upsample(inputs, filters, out_size, layer_no):
    name = "dconv_layer_{0}.{1}".format(version, layer_no)

    layer = tf.layers.conv2d_transpose(inputs, filters, (5,5), padding='same', name=name, activation=tf.nn.relu)

    kernel = gr.get_tensor_by_name('{0}/kernel:0'.format(name))
    bias = gr.get_tensor_by_name('{0}/bias:0'.format(name))
    reset_layers.append([kernel, bias])

    return tf.image.resize_images(layer, size=(out_size,out_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  layer = upsample(layer, 512, 16, 1)
  layer = upsample(layer, 512, 32, 2)
  layer = upsample(layer, 512, 64, 3)

  layer = tf.layers.conv2d_transpose(layer, 3, (5,5), activation=tf.sigmoid, padding='same')

  return layer

def Categorizer(layer):
  return tf.layers.dense(layer, 2)

def AutoEncoder():
  model_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="signal_input")
  model_output = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="signal_output")
  reset_layers_A = []
  encoder = Encoder(model_input)
  decoder_A = Decoder(encoder, "A", reset_layers_A)
  reset_layers_B = []
  decoder_B = Decoder(encoder, "B", reset_layers_B)
  categorizer = Categorizer(encoder)
  return decoder_A, decoder_B, model_input, model_output, categorizer, reset_layers_A, reset_layers_B

autoencoder_A, autoencoder_B, model_input, model_output, model_categorizer, reset_layers_A, reset_layers_B = AutoEncoder()
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
  def predict(images, indexes, autoencoder):
    placeholders ={ 
        model_input: images
        }
    return sess.run(autoencoder, placeholders)

  def predict_cat(images):
    placeholders ={ 
        model_input: images
    }
    one_hots = sess.run(model_prob_cat, placeholders)
    preds = np.argmax(one_hots, axis=1)
    return [ ['CAGE', 'TRUMP'][pred] for pred in preds ]

  A_images = [warp_image(cage_images[indexes_A[i]])[1] for i in range(len(cage_images))]
  raw_cage = A_images[0:4]
  decoded_cage = predict(A_images[0:4], indexes_A, autoencoder_A)
  decoded_cage_as_trump = predict(A_images[0:4], indexes_A, autoencoder_B)
  cage_predict = predict_cat(A_images[0:4])

  B_images = [warp_image(trump_images[indexes_B[i]])[1] for i in range(len(trump_images))]
  raw_trump= B_images[0:4]
  decoded_trump = predict(B_images[0:4], indexes_B, autoencoder_B)
  decoded_trump_as_cage = predict(B_images[0:4], indexes_B, autoencoder_A)
  trump_predict = predict_cat(B_images[0:4])

  def no_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  def plot_image(col, image, cat):
    ax = plt.subplot(n_rows,n_cols,(4*col)+i+1)
    plt.imshow(image) 
    ax.text(70, 50, cat, fontsize=15, backgroundcolor='w')
    no_axis(ax)

  for i in range(n_images):
    plot_image(0, raw_cage[i], cage_predict[i])
    plot_image(1, decoded_cage[i], cage_predict[i])
    plot_image(2, decoded_cage_as_trump[i], cage_predict[i])
    plot_image(3, raw_trump[i], trump_predict[i])
    plot_image(4, decoded_trump[i], trump_predict[i])
    plot_image(5, decoded_trump_as_cage[i], trump_predict[i])

  plt.pause(0.001)

saver = tf.train.Saver()

model_prob_cat = tf.nn.softmax(model_categorizer)
model_output_cat = tf.placeholder(tf.float32, shape=[None, 2], name="model_output_cat")
model_loss_cat = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_output_cat, logits=model_categorizer))
model_optimizer_cat = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op_cat = model_optimizer_cat.minimize(model_loss_cat)

model_loss_A = tf.reduce_mean(tf.keras.losses.mean_absolute_error(tf.layers.flatten(model_output), tf.layers.flatten(autoencoder_A)))
model_optimizer_A = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op_A = model_optimizer_A.minimize(model_loss_A)

model_loss_B = tf.reduce_mean(tf.keras.losses.mean_absolute_error(tf.layers.flatten(model_output), tf.layers.flatten(autoencoder_B)))
model_optimizer_B = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op_B = model_optimizer_B.minimize(model_loss_B)

combine_images = False

images_A = cage_images
indexes_A = [i for i in range(len(images_A))]
onehot_A = [[1,0] for _ in range(BATCH_SIZE)]

images_B = trump_images
indexes_B = [i for i in range(len(images_B))]
onehot_B = [[0,1] for _ in range(BATCH_SIZE)]

def get_batch(indexes, start, end, x):
  value = [x[indexes[i]] for i in range(start, end)]
  return value

epochs = 10000
steps = 50
batches = min(len(images_A), len(images_B)) // BATCH_SIZE

sess = tf.Session()

sess.run(tf.global_variables_initializer())

saved_layers_A = sess.run(reset_layers_A)
saved_layers_B = sess.run(reset_layers_B)
def reset_layers(resets, saves, keep_n):
  for i in range(len(resets)):
    if i >= keep_n:
      reset = resets[i]
      save = saves[i]
      for j in range(2):
        sess.run(tf.assign(reset[j], save[j]))
 
#pdb.set_trace()
#reset_layers(reset_layers_A, saved_layers_A, 0)
#pdb.set_trace()
   
saved_model_path = tf.train.latest_checkpoint(SAVE_DIR)
if False and saved_model_path:
  saver.restore(sess, saved_model_path)

show_graph(sess)

last_time = time.time()
reset_at_layer = 0
for epoch in range(epochs):
  print("\nEpoch {0} seconds {1}".format(epoch, time.time()-last_time))
  last_time = time.time()
  random.shuffle(indexes_A)
  random.shuffle(indexes_B)

  if (epoch % 9) == 1:
    pdb.set_trace()
    print("Reseting layer {0}".format(reset_at_layer))
    reset_layers(reset_layers_A, saved_layers_A, reset_at_layer)
    reset_layers(reset_layers_B, saved_layers_B, reset_at_layer)
    reset_at_layer += 1

  for step in range(steps):
    for batch in range(batches):
      start = batch*BATCH_SIZE
      end = (batch+1)*BATCH_SIZE

      batch_A = get_batch(indexes_A, start, end, images_A)
      inputs_A, outputs_A = warp_images(batch_A)
      placeholders = {
        model_input: inputs_A,
        model_output: outputs_A,
        model_output_cat: onehot_A
      }
      loss_A, loss_cat_A, _, _ = sess.run([model_loss_A, model_loss_cat, model_train_op_A, model_train_op_cat], placeholders)
      '''
      placeholders = {
        model_input: get_batch(indexes_A, start, end, timages_A),
        model_output_cat: onehot_A
      }
      loss_cat_A, _ = sess.run([model_loss_cat, model_train_op_cat], placeholders)
      '''
      batch_B = get_batch(indexes_B, start, end, images_B)
      inputs_B, outputs_B  = warp_images(batch_B)
      placeholders = {
        model_input: inputs_B, 
        model_output: outputs_B,
        model_output_cat: onehot_B
      }
      loss_B, loss_cat_B, _, _ = sess.run([model_loss_B, model_loss_cat, model_train_op_B, model_train_op_cat], placeholders)
      '''
      placeholders = {
        model_input: get_batch(indexes_B, start, end, timages_B),
        model_output_cat: onehot_B
      }
      loss_cat_B, _ = sess.run([model_loss_cat, model_train_op_cat], placeholders)
      '''

    print("Step: {0} loss_a: {1}/{2} loss_b: {3}/{4}\r".format(step, loss_A, loss_cat_A, loss_B, loss_cat_B))
    show_graph(sess)
    saver.save(sess, SAVE_FILE)
