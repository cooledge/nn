import tensorflow as tf
import numpy as np
import cv2
import time
import math
import pdb
import fnmatch
import time
import urllib
import os.path
import random
import argparse
from collections import defaultdict
from collections import deque
# from moving.chw2hwc import chw2hwc
from PIL import Image
import argparse
import glob
import sys
import matplotlib.pyplot as plt
from itertools import product

parent_dir = os.path.dirname(os.path.abspath(__file__)) + "/../target"
sys.path.append(parent_dir)

target_graph = tf.Graph()
with target_graph.as_default():
  from target import Predict as Target_Predict

parent_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(parent_dir)
import utils
import params

# 46X64X1
def show_image_array(ia):
  print("--------------------+-------------------+---------------------+-------------------+-------------------+")
  ia = np.array(ia).astype(int)
  #for r in range(46):
  for r in range(1):
    for c in range(64):
      sys.stdout.write("{0:1} ".format(ia[r][c][0]))
    print("")
           
target_predict = Target_Predict()

def image_to_logits(image):
  return target_predict.logits(image)

def image_to_target(image):
  decorated = np.array(image)
  target_predict.run(image, decorated)
  return decorated

#batch_size = 64
batch_size = 2
n_features_last_layer = 3

model_dir = os.path.dirname(os.path.abspath(__file__)) + "/model"

if not os.path.exists(model_dir):
  os.makedirs(model_dir)

model_filename = model_dir + "/model"

actions = ['f', 'b', 'l', 'r', 's']

def action_to_one_hot_index(action):
  return actions.index(action)

n_actions = len(actions)
# number of actions to be generated
cell_state_size = 128
rnn_cell_depth = 3
n_len = params.N_ACTIONS
#n_len = 1
n_scale = 5
n_rows = int(480 / n_scale)
n_cols = int(640 / n_scale)

elements = [action for action in params.ACTIONS if not action == 's']
if 's' in params.ACTIONS:
  index_to_dir = ['s']
else:
  index_to_dir = []
for i in range(params.N_ACTIONS):
  index_to_dir += ["".join(str) for str in product(elements, repeat=i+1)]

dir_to_index = {}
for i, choice in enumerate(index_to_dir):
  dir_to_index[choice] = i

def ind2cat(ind):
  return index_to_dir[ind]

n_directions = len(dir_to_index)

def actions_to_canonical(actions):
  n_actions = len(actions)
  moves = [action for action in actions if action != 's']
  reduced = []
  n_moves = len(moves)
  i = 0
  while i < n_moves-1:
    if moves[i] == 'f' and moves[i+1] == 'b':
      i += 2
    elif moves[i] == 'b' and moves[i+1] == 'f':
      i += 2
    else:
      reduced.append(moves[i])
      i += 1

  if i < n_moves:
    reduced.append(moves[i])

  return ''.join(reduced)

assert actions_to_canonical('bls') == 'bl'
assert actions_to_canonical('blsbf') == 'bl'
assert actions_to_canonical('bffss') == 'f'
assert actions_to_canonical('fsfls') == 'ffl'
assert actions_to_canonical('fbfls') == 'fl'

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def get_prefix_fn(filename):
  return filename[:filename.find('-')]

def get_counter_fn(filename):
  return int(filename[filename.find('-')+1:filename.find('.')])

def sort_key(filename):
  return (get_prefix_fn(filename), get_counter_fn(filename))

def get_files(dir):
  files = [file for file in os.listdir(dir) if fnmatch.fnmatch(file, '*.jpg')]
  files = sorted(files, key=sort_key)
  return files

n_files = 2

def filepath(data_dir, filename):
  return "{0}/{1}".format(data_dir, filename)

def get_counter(files, i):
  if i >= len(files):
    return -1
  filename = files[i]
  return get_counter_fn(filename)

def si(image, title=""):
  cv2.imshow(title, image)
  cv2.waitKey(1)

def load_image(data_dir, filename):
  image = utils.load_image(data_dir, filename, n_rows, n_cols)
  logits = np.array(image_to_logits(image))
  
  max = np.amax(logits)
  scale = 255.0/max
  logits *= scale

  # save it
  path = '/home/dev/code/nn/robotcar/movement/ldata/{0}'
  cv2.imwrite(path.format(filename), logits)
  #pdb.set_trace()
  #si(logits)
  '''
  show_logits = np.array(logits)
  for r in range(show_logits.shape[0]):
    for c in range(show_logits.shape[1]):
      if logits[r][c] > 0.0:
        show_logits[r][c] = 255*logits[r][c]

  # 'bsfrf_20180605061616-1.jpg'
  si(image_to_target(image), "target")
  si(image, "image")
  si(show_logits, "logits")
  pdb.set_trace()
  '''
  return logits

categories = ['forward', 'backward', 'left', 'right', 'stop']

def get_actions(files, batch_no):
  file = files[batch_no]
  actions = file[:file.find('_')]
  actions = actions_to_canonical(actions)
  if actions == '':
    actions = 's'
  idx = dir_to_index[actions]
  one_hots = np.zeros(n_directions)
  one_hots[idx] = 1
  return one_hots

def set_actions(files, directions, batch_no):
  actions = get_actions(files, batch_no)
  for action_no in range(n_len):
    directions[batch_no][action_to_one_hot_index(actions[action_no])] = 1

def plot_figures(figures, nrows = 1, ncols=1):
  """
  Plot a dictionary of figures.

  Parameters
  ----------
  figures : <title, figure> dictionary
  ncols : number of columns of subplots wanted in the display
  nrows : number of rows of subplots wanted in the figure
  """
  fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
  for ind,title in enumerate(figures):
    axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
    axeslist.ravel()[ind].set_title(title)
    axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
  plt.show()

def get_image_sequence(data_dir, files, i):
  if get_counter(files, i) == 1:
    images = [load_image(data_dir, files[i+j]) for j in range(n_files)]
    
    #plot_figures({'a':images[0], 'b':images[1]}, 1, 2)
    #pdb.set_trace()
   
    return images

def index2action(index):
  return actions[int(index)]

for index in range(len(actions)):
  assert index2action(index) == actions[action_to_one_hot_index(actions[index])]
 
def indexes2actions(indexes):
  return ''.join([index2action(index) for index in indexes])

assert indexes2actions([0,1,2,3,4]) == 'fblrs'

def batch_indexes2actions(batches):
  return [indexes2actions(batch) for batch in batches]

def action2one_hot(action):
  one_hots = np.zeros((n_actions))
  one_hots[action_to_one_hot_index(action)] = 1
  return one_hots

def actions2one_hots(actions):
  return [action2one_hot(action) for action in actions]

def batch_of_actions2one_hots(batches_of_actions):
  return np.array([actions2one_hots(actions) for actions in batches_of_actions])
  
data_dir = './data'
percent_test = 0.1

def split_to_training_and_test_data():
  files = get_files(data_dir)
  prefix_to_files = {}
  for filename in files:
    prefix = get_prefix_fn(filename)
    if not prefix in prefix_to_files:
      prefix_to_files[prefix] = []
    prefix_to_files[prefix].append(filename)
  keys = list(prefix_to_files.keys())
  random.shuffle(keys)
  n_test = int(percent_test*len(keys))
  test_prefix = keys[:n_test]
  training_prefix = keys[n_test:]

  training_files = []
  for prefix in training_prefix:
    training_files.extend(prefix_to_files[prefix])

  test_files = []
  for prefix in test_prefix:
    test_files.extend(prefix_to_files[prefix])

  assert len(training_files)+len(test_files) == len(files)
  return training_files, test_files

training_files, test_files = split_to_training_and_test_data()

def load_test_data():
  files = test_files

  # take N images in a row and make them each a channel in a single
  images = []
  directions = []
  for i in range(len(files)):
    image_sequence = get_image_sequence(data_dir, files, i)
    if image_sequence:
      images.append(image_sequence)
      directions.append(get_actions(files, i))

  return np.array(images), np.array(directions)

def load_training_data():
  files = training_files

  # take N images in a row and make them each a channel in a single
  images = []
  directions = []
  for i in range(len(files)):
    image_sequence = get_image_sequence(data_dir, files, i)
    if image_sequence:
      images.append(image_sequence)
      directions.append(get_actions(files, i))

  indexes = [i for i in range(len(images))]
  random.shuffle(indexes)
  simages = [images[indexes[i]] for i in range(len(images))]
  #sdirections = [directions[indexes[i]] for i in range(len(directions))] 
  images = simages
  #directions = sdirections

  split_at = int(len(images)*0.9)
  X_train = np.array(images[:split_at])
  Y_train = directions[:split_at]

  X_test = np.array(images[split_at:])
  Y_test = directions[split_at:]

  return X_train, Y_train, X_test, Y_test

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# number of classes

def tf_add_conv(layer, filters, kernel_size=[5,5], name=None, reuse=False, include_pool=True, pool_size=[2,2]):
  layer = tf.layers.conv2d(inputs=layer, filters=filters, kernel_size=kernel_size, padding='same', activation=tf.nn.relu, name=name, reuse=reuse)
  if include_pool:
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=pool_size, strides=pool_size[0], padding='same')
  return layer

# get some shared features on the images with reuse
def tf_add_conv_by_image(layer, n_filters, name, kernel_size=[5,5], include_pool=True, pool_size=[2,2]):
  layers = tf.split(layer, n_files, axis=1)
  nps = layers[0].shape.as_list()
  nps.remove(1)
  layers = [tf.reshape(layer, (-1, nps[1], nps[2], nps[3])) for layer in layers]
  layers = [tf_add_conv(layer, n_filters, name=name, kernel_size=kernel_size, reuse=(idx!=0), include_pool=include_pool, pool_size=pool_size) for idx, layer in enumerate(layers)]
  n_rows = layers[0].shape[1]
  n_cols = layers[0].shape[2]
  layers = [tf.reshape(layer, (-1, 1, n_rows, n_cols, n_filters)) for layer in layers]
  layer = tf.concat(layers, 1)
  return layer

model_input = tf.placeholder(tf.float32, shape=(None, n_files, n_rows, n_cols), name='model_input')
model_batch_size = tf.placeholder(tf.int32, [], name='batch_size')
model_keep_prob = tf.Variable(0.25, dtype=tf.float32)

def Model3D_CNN(model_input):
  layer = tf.reshape(model_input, (-1, n_files, n_rows, n_cols, 1))

  layer = tf_add_conv_by_image(layer, 64, 'layer1', kernel_size=[2,2], include_pool=False)
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf_add_conv_by_image(layer, 64, 'layer2', kernel_size=[5,5])


  pdb.set_trace() 
  rnn_inputs = tf.split(layer, n_len, axis=1)
  
  #rnn_inputs = [layer for _ in range(n_len)]

  def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(cell_state_size)
  
  model_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(rnn_cell_depth)])
  #model_initial_state = model_rnn_cell.zero_state(model_batch_size, tf.float32) 
  model_initial_state = model_rnn_cell.zero_state(model_batch_size, tf.float32)
  model_rnn_outputs, model_rnn_state = tf.nn.static_rnn(model_rnn_cell, rnn_inputs, model_initial_state)
  pdb.set_trace()
  layers = [tf.layers.dense(rnn_output, len(actions)) for rnn_output in model_rnn_outputs]




  #layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.flatten(layer)
  #layer = tf.layers.dense(layer, 256, activation=tf.nn.relu)
  layer = tf.layers.dense(layer, 128, activation=tf.nn.relu)
  #layer = tf.layers.dense(layer, 64, activation=tf.nn.relu)
  layer = tf.layers.dense(layer, n_directions, activation=tf.nn.relu)
 
  model_logits = layer
  return model_logits

def Model3D(model_input):
  layer = tf.reshape(model_input, (-1, n_files, n_rows, n_cols, 1))

  layer = tf_add_conv_by_image(layer, 64, 'layer1', kernel_size=[2,2], include_pool=False)
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf_add_conv_by_image(layer, 12, 'layer2', kernel_size=[5,5])
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  #layer = tf_add_conv_by_image(layer, 32, 'layer3', kernel_size=[5,5])
  #layer = tf_add_conv_by_image(layer, 32, 'layer2', include_pool=False)
  #layer = tf_add_conv_by_image(layer, 64, 'layer3', include_pool=False)

  # TRY DROPOUTS in differen layers 
  # TRY DIFFEReNT DROPouT PROB
  # try making sd of the normal smaller
  # TRY Check circles -> are they all the same size -> use that info -> pool up to the right size
  layer = tf.layers.conv3d(inputs=layer, filters=16, kernel_size=2, padding='same', activation=tf.nn.relu)
  #layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[2,2,2], strides=2, padding='same')
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.flatten(layer)
  #layer = tf.layers.dense(layer, 256, activation=tf.nn.relu)
  layer = tf.layers.dense(layer, 128)
  #layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  #layer = tf.layers.dense(layer, 64, activation=tf.nn.relu)
  layer = tf.layers.dense(layer, n_directions)
  model_logits = layer
  return model_logits

def Model3D_2(model_input):
  layer = tf.reshape(model_input, (-1, n_files, n_rows, n_cols, 1))
  layer = tf.layers.conv3d(inputs=layer, filters=nb_filters, kernel_size=n_files, padding='same', activation=tf.nn.relu)
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.conv3d(inputs=layer, filters=nb_filters, kernel_size=n_files, padding='same', activation=tf.nn.relu)
  layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[4,2,2], strides=2, padding='same')
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[4,4,4], strides=2, padding='same')
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.flatten(layer)
  layer = tf.layers.dense(layer, 128, activation=tf.nn.relu)
  layer = tf.layers.dense(layer, n_directions)
  return layer

def Model3D_3(model_input):
  layer = tf.reshape(model_input, (-1, n_files, n_rows, n_cols, 1))
#continue wiht reductin the filter
  layer = tf_add_conv_by_image(layer, 32, 'layer1', kernel_size=[10,10], include_pool=False)
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf_add_conv_by_image(layer, 64, 'layer2', kernel_size=[10,10], pool_size=[5,5])
  #layer = tf_add_conv_by_image(layer, 32, 'layer3', kernel_size=[5,5])
  #layer = tf_add_conv_by_image(layer, 32, 'layer2', include_pool=False)
  #layer = tf_add_conv_by_image(layer, 64, 'layer3', include_pool=False)

  # TRY DROPOUTS in differen layers 
  # TRY DIFFEReNT DROPouT PROB
  # TRY Check circles -> are they all the same size -> use that info -> pool up to the right size
  layer = tf.layers.conv3d(inputs=layer, filters=16, kernel_size=2, padding='same', activation=tf.nn.relu)
  #layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[2,2,2], strides=2, padding='same')
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.flatten(layer)
  #layer = tf.layers.dense(layer, 256, activation=tf.nn.relu)
  layer = tf.layers.dense(layer, 128)
  #layer = tf.layers.dense(layer, 64, activation=tf.nn.relu)
  layer = tf.layers.dense(layer, n_directions)
 
  model_logits = layer
  return model_logits

def Model3D_LR(model_input):
  layer = tf.reshape(model_input, (-1, n_files, n_rows, n_cols, 1))

  if False:
    layer = tf_add_conv_by_image(layer, 64, 'layer1', kernel_size=[10,10], include_pool=False)
    layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
    #n_features_last_layer = 52
    layer = tf_add_conv_by_image(layer, n_features_last_layer, 'layer2', kernel_size=[5,5])
  else:
    layer = tf_add_conv_by_image(layer, n_features_last_layer, 'layer1', kernel_size=[10,10])
    #layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)

  # ?, 2 , 48, 64, 1
 
  if n_features_last_layer == 1: 
    fake = np.zeros((batch_size, 2, 48, 64, n_features_last_layer))
    fake[0][0][0][32][0] = 2.0
    fake[0][0][0][31][0] = 1.9
    fake[0][1][0][56][0] = 2.0

    if batch_size >= 2:
      fake[1][0][0][32][0] = 2.0
      fake[1][0][0][31][0] = 1.9
      fake[1][1][0][56][0] = 2.0

    if batch_size >= 3:
      fake[2][0][0][32][0] = 2.0
      fake[2][0][0][31][0] = 1.9
      fake[2][1][0][56][0] = 2.0

    layer = tf.constant(fake, dtype=tf.float32)
  elif n_features_last_layer == 2: 
    fake = np.zeros((batch_size, 2, 48, 64, n_features_last_layer))
    # fake[batch][page][row][col][feature]
    fake[0][0][0][32][0] = 2.0
    fake[0][0][0][31][0] = 1.9
    fake[0][1][0][56][0] = 2.0

    fake[0][0][0][12][1] = 1.0
    fake[0][0][0][21][1] = 0.9
    fake[0][1][0][36][1] = 1.0
   
    if batch_size >= 2: 
      fake[1][0][0][32][0] = 2.0
      fake[1][0][0][31][0] = 1.9
      fake[1][1][0][56][0] = 2.0

      fake[1][0][0][12][1] = 1.0
      fake[1][0][0][21][1] = 0.9
      fake[1][1][0][36][1] = 1.0
    layer = tf.constant(fake, dtype=tf.float32)
  elif n_features_last_layer == 3: 
    fake = np.zeros((batch_size, 2, 48, 64, n_features_last_layer))
    fake[0][0][0][32][0] = 2.0
    fake[0][0][0][31][0] = 1.9
    fake[0][1][0][56][0] = 2.0

    fake[0][0][0][12][1] = 1.0
    fake[0][0][0][21][1] = 0.9
    fake[0][1][0][36][1] = 1.0
    
    fake[0][0][0][1][2] = 1.0
    fake[0][0][0][10][2] = 0.9
    fake[0][1][0][60][2] = 1.0
   
    if batch_size >= 2: 
      fake[1][0][0][32][0] = 2.0
      fake[1][0][0][31][0] = 1.9
      fake[1][1][0][56][0] = 2.0

      fake[1][0][0][12][1] = 1.0
      fake[1][0][0][21][1] = 0.9
      fake[1][1][0][36][1] = 1.0

      fake[1][0][0][1][2] = 1.0
      fake[1][0][0][10][2] = 0.9
      fake[1][1][0][60][2] = 1.0
   
    layer = tf.constant(fake, dtype=tf.float32)
  
  # ([Dimension(None), Dimension(2), Dimension(48), Dimension(64), Dimension(2)]
  layers = tf.split(layer, 2, axis=1)
  layers = [tf.squeeze(layer, axis=1) for layer in layers]
 
  pages = [tf.split(layer, n_features_last_layer, axis=3) for layer in layers]
  #pages = [[tf.nn.softmax(layer) for layer in page] for page in pages]
  #model_coolness = pages
  pages = [[tf.layers.max_pooling2d(inputs=layer, pool_size=[n_rows/2, 1], strides=1, padding='valid') for layer in page] for page in pages]
  pages = [[(layer / (layer+0.0000001)) for layer in page] for page in pages] 

  scale = tf.placeholder(tf.float32, shape=pages[0][0].shape, name='model_scale')
  layers = [layer * scale for layer in layers]
  pages = [[layer * scale for layer in page] for page in pages]

  def get_coolness(layer, scale):
    layers = tf.split(layer, 2, axis=1)
    layers = [tf.squeeze(layer, axis=1) for layer in layers]
   
    pages = [tf.split(layer, 1, axis=3) for layer in layers]
    #pages = [[tf.nn.softmax(layer) for layer in page] for page in pages]
    #model_coolness = pages
    pages = [[tf.layers.max_pooling2d(inputs=layer, pool_size=[n_rows/2, 1], strides=1, padding='valid') for layer in page] for page in pages]
    pages = [[(layer / (layer+0.0000001)) for layer in page] for page in pages] 

    #scale = tf.placeholder(tf.float32, shape=pages[0][0].shape, name='model_scale')
    layers = [layer * scale for layer in layers]
    pages = [[layer * scale for layer in page] for page in pages]
    #pdb.set_trace()
    # (batch, page, features, ? row, col, ?)
    # (2, 2, 1, 1, 1, 64, 1)
    # (2, 2, 2, 1, 1, 64, 1)
    model_coolness = pages

    pages2 = [[tf.squeeze(layer, axis=3) for layer in page] for page in pages]
    pages2 = [[layer - tf.reduce_max(layer, axis=2) for layer in page] for page in pages2]
    pages2 = [[(layer / (layer-0.0000001))*-2+1 for layer in page] for page in pages2]
    pages2 = [[tf.nn.relu(layer) for layer in page] for page in pages2]
    
    pages2 = [[tf.reshape(layer, (-1, 1, 64, 1)) for layer in page] for page in pages2]
    pages2 = [[layer * scale for layer in page] for page in pages2]
    #pdb.set_trace()
    pages2 = [[tf.reshape(layer, (-1, 64)) for layer in page] for page in pages2]
    #pdb.set_trace()
    #pages2 = [[tf.reduce_sum(layer, axis=1) for layer in page] for page in pages2]
    #diffs = [l-r for l,r in zip(pages2[0], pages2[1])]
    #model_coolness = diffs
    #model_coolness = pages2
    return model_coolness

  pdb.set_trace() 
  scale = tf.placeholder(tf.float32, shape=(1,1,64,1), name='model_scale')
  # n=1,f=1 [<tf.Tensor 'split_4:0' shape=(1, 2, 48, 64, 1) dtype=float32>]
  # n=1,f=2 [<tf.Tensor 'split_4:0' shape=(?, 2, 48, 64, 2) dtype=float32>] 
  # n=2,f=1 [<tf.Tensor 'split_4:0' shape=(1, 2, 48, 64, 1) dtype=float32>, <tf.Tensor 'split_4:1' shape=(1, 2, 48, 64, 1) dtype=float32>]
  # n=2,f=2 [<tf.Tensor 'split_4:0' shape=(?, 2, 48, 64, 2) dtype=float32>, <tf.Tensor 'split_4:1' shape=(?, 2, 48, 64, 2) dtype=float32>]

  batch = tf.split(layer, batch_size, axis=0)
  batch_by_features = [tf.split(element, n_features_last_layer, axis=4) for element in batch]
  # n =1 [[[<tf.Tensor 'mul_6:0' shape=(1, 1, 64, 1) dtype=float32>], [<tf.Tensor 'mul_7:0' shape=(1, 1, 64, 1) dtype=float32>]]]
  # n= 2 [[[<tf.Tensor 'mul_6:0' shape=(1, 1, 64, 1) dtype=float32>], [<tf.Tensor 'mul_7:0' shape=(1, 1, 64, 1) dtype=float32>]], [[<tf.Tensor 'mul_14:0' shape=(1, 1, 64, 1) dtype=float32>], [<tf.Tensor 'mul_15:0' shape=(1, 1, 64, 1) dtype=float32>]]]
  model_coolness= [[get_coolness(layer, scale) for layer in features] for features in batch_by_features]
  #model_coolness = batch_by_features
  # [[<tf.Tensor 'mul_6:0' shape=(2, 1, 64, 1) dtype=float32>], [<tf.Tensor 'mul_7:0' shape=(2, 1, 64, 1) dtype=float32>]]
  #model_coolness= get_coolness(layer, scale)
  #model_coolness = tf.reshape(tf.concat([tf.concat(single, 0) for single in model_coolness], 0), (-1, 2))
  #model_coolness = tf.layers.dense(model_coolness, 2)

  layers = [tf.concat(page, axis=3) for page in pages]
  # (?, 1, 64, 52) 
  layers = [tf.squeeze(layer, axis=1) for layer in layers]
  
  # (?, 64, 52) 
  n_filters = layers[0].shape[2]

  def dude(layer):
    layer = tf.squeeze(layer, axis=2)
    return tf.reduce_sum(tf.nn.softmax(layer*100)*layer, axis=1)

  layers1 = [dude(layer) for layer in tf.split(layers[0], n_filters, axis=2)]
  layers2 = [dude(layer) for layer in tf.split(layers[1], n_filters, axis=2)]
  #model_coolness = [layers1, layers2]


  lefts = [layer1-layer2 for layer1,layer2 in zip(layers1, layers2)]
  rights = [layer2-layer1 for layer1,layer2 in zip(layers1, layers2)]
  lefts = [tf.reshape(left, (-1, 1)) for left in lefts] 
  left = tf.concat(lefts, axis=1)
  left = tf.reshape(left, (-1, 1, n_features_last_layer))
  rights = [tf.reshape(right, (-1, 1)) for right in rights] 
  # (?, 52)
  right = tf.concat(rights, axis=1)
  right = tf.reshape(right, (-1, 1, n_features_last_layer))
  # (?, 2, 52)
  layer = tf.concat([left, right], axis=1)
  #model_coolness = layer

  #layer = tf.layers.dense(layer, 1, activation=tf.nn.relu)
  layer = tf.layers.dense(layer, 1)
  layer = tf.squeeze(layer, 2)
  pdb.set_trace()
  model_logits = layer
  return model_logits, scale, model_coolness

#train the target to a smaller standard deviation
# raise any non-zero pixel to full 255
model_logits, model_scale, model_coolness = Model3D_LR(model_input)
#model_softmax = tf.nn.softmax(model_logits)
#model_predict = tf.argmax(model_softmax, axis=1)
model_outputs = tf.placeholder(tf.int32, shape=[None, n_directions], name="model_outputs")

model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_outputs, logits=model_logits))
#model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=model_outputs, logits=tf.stop_gradient(model_logits)))
model_predict = tf.nn.softmax(model_logits)

#model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_optimizer = tf.train.AdamOptimizer()
model_train_op = model_optimizer.minimize(model_loss)
model_accuracy = tf.metrics.accuracy(model_outputs, model_predict)

session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

saver = tf.train.Saver()

class Predict:

  def __init__(self):
    self.images = deque()

  def run(self, image, display_image):
    from arrow import draw_arrow
    image = cv2.resize(image, (n_cols, n_rows))
    self.images.append(image)
    category = None
    if len(self.images) == n_files: 
      try:
        predict = session.run(model_predict, { model_input: [self.images], model_keep_prob: 1.0 } )
        category = ind2cat(predict[0])
        draw_arrow(display_image, category)
      except ValueError:
        category = None
      self.images.popleft()

    return category

if not __name__ == '__main__':
  try:
    saver.restore(session, model_filename)
  except Exception as e:
    0 # ignore 

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Detect direction of motion")

  parser.add_argument("--epochs", default=500, type=int)
  parser.add_argument("--train", dest='train', default=True, action='store_true')
  parser.add_argument("--no-train", dest='train', action='store_false')
  parser.add_argument("--test", dest='test', default=True, action='store_true')
  parser.add_argument("--no-test", dest='test', action='store_false')
  parser.add_argument("--clean", dest='clean', default=False, action='store_true')
  parser.add_argument("--test-predict", dest='test_predict', default=False, type=bool)

  args = parser.parse_args()

  if not args.clean:
    try:
      saver.restore(session, model_filename)
    except Exception as e:
      0 # ignore 

  if args.clean:
    os.system("rm {0}/*".format(model_dir))

  n_epochs = args.epochs

  X_train, Y_train, X_validation, Y_validation = load_training_data()

  def one_hots_to_directions(one_hots):
    return [index_to_dir[np.argmax(one_hot)] for one_hot in one_hots]

  '''
  def get_scale(batch_size):
    scale = np.zeros((1, 64, n_features_last_layer))
    for f in range(n_features_last_layer):
      i = 1
      for c in range(64):
        scale[0][c][f] = i
        i += 1
    return scale
  '''

  def get_scale(batch_size):
    scale = np.zeros((1, 1, 64, 1))
    i = 1
    for c in range(64):
      scale[0][0][c][0] = i
      i += 1
    return scale


  def get_accuracy(X, Y): 
    n_batch_size = 32
    n_batches = int(X.shape[0] / n_batch_size)
    predictions = []
    total_loss = 0.0
    for i in range(n_batches):
      start = i*n_batch_size
      end = start + n_batch_size
      
      placeholders = { 
          model_batch_size: n_batch_size,  
          model_input: X[start:end], 
          model_outputs: Y[start:end], 
          model_scale: get_scale(n_batch_size),
          model_keep_prob: 1.0 
      }
      #pdb.set_trace()
      batch_predictions, batch_loss, coolness = session.run([model_predict, model_loss, model_coolness], placeholders)
      
      total_loss += batch_loss
      predictions += one_hots_to_directions(batch_predictions)
    loss = total_loss / n_batches
    labels = one_hots_to_directions(Y)
    right = len([True for (label, prediction) in zip(labels, predictions) if label == prediction])
    accuracy = float(right) / float(len(predictions)) 
    return accuracy, predictions, labels, loss

  if args.train:
    nb_train_samples, img_channels, n_rows, n_cols = X_train.shape
    nb_validation_samples = X_validation.shape[0]

    print(nb_train_samples, 'train samples')
    print(nb_validation_samples, 'validation samples')

    for epoch in range(n_epochs):
      n_batches = nb_train_samples // batch_size
      indexes = list(range(nb_train_samples))
      random.shuffle(indexes)

      def select(elements, indexes, start, end):
        return [elements[indexes[i]] for i in range(start, end)] 

      for batch_no in range(n_batches):
        start = batch_no * batch_size
        end = start + batch_size
        batch_input = select(X_train, indexes, start, end)
        batch_output = select(Y_train, indexes, start, end)
        #lds = session.run(layers_dude, { model_batch_size: batch_size, model_input: batch_input, model_outputs: batch_output })
        #ss = [ld.shape for ld in lds]
        '''
        loss_before = session.run(model_loss, { model_batch_size: batch_size, model_input: batch_input, model_outputs: batch_output })
        model_gradients = []
        for (gradient, _) in model_optimizer.compute_gradients(model_loss):
          if not gradient is None:
            model_gradients.append(gradient)
        pdb.set_trace()
        gradients = session.run(model_gradients, { model_batch_size: batch_size, model_input: batch_input, model_outputs: batch_output })
        '''
        placeholder = { 
          model_batch_size: batch_size, 
          model_input: batch_input, 
          model_outputs: batch_output,
          model_scale: get_scale(batch_size)
        }
        coolness, scale = np.array(session.run([model_coolness, model_scale], placeholder))
        pdb.set_trace()
        '''
        for b in range(batch_size):
          for f in range(n_features_last_layer):
            for page in range(2):
              show_image_array(coolness[b][f][0][page])
        '''
        for b in range(batch_size):
          for f in range(n_features_last_layer):
            for page in range(2):
              show_image_array(coolness[b][f][page][0][0])
        #loss, _ = session.run([model_loss, model_train_op], placeholder) 
        '''
        loss_after = session.run(model_loss, { model_batch_size: batch_size, model_input: batch_input, model_outputs: batch_output })
        pdb.set_trace()
        pdb.set_trace()
        '''
      accuracy, predictions, _, _ = get_accuracy(X_validation, Y_validation)
      accuracy_train, predictions_train, _, tloss  = get_accuracy(X_train[0:32], Y_train[0:32])
      print("Epoch {0} Validation Accuracy: {1} Train Accuracy {2} TLoss {3}".format(epoch, accuracy, accuracy_train, tloss))
      #print("      tests {0}\n      predictions: {1}".format(Y_validation, predictions))
      #print("      tests {0} predictions: {1}".format(Y_train, predictions))
      saver.save(session, model_filename)
     
  if args.test:
    X_test, Y_test = load_test_data()
    accuracy, predictions, labels, _ = get_accuracy(X_test, Y_test)
    print("Test/Predictions")
    for t, p in zip(labels, predictions):
      print("{0}-{1}".format(t, p))
    #print("test/predictions: {0}".format(zip(Y_test, predictions)))
    print("Test Accuracy: {0}".format(accuracy))


