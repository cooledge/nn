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

parent_dir = os.path.dirname(os.path.abspath(__file__)) + "/../target"
sys.path.append(parent_dir)

target_graph = tf.Graph()
with target_graph.as_default():
  from target import Predict as Target_Predict

parent_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(parent_dir)
import utils
import params

target_predict = Target_Predict()

def image_to_logits(image):
  return target_predict.logits(image)

def image_to_target(image):
  decorated = np.array(image)
  target_predict.run(image, decorated)
  return decorated

#batch_size = 64
batch_size = 8
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

def actions_to_canonical(actions):
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

  return ''.join(reduced) + 's'*(n_actions-len(reduced))

assert actions_to_canonical('blsbf') == 'blsss'
assert actions_to_canonical('bffss') == 'fssss'
assert actions_to_canonical('fsfls') == 'fflss'
assert actions_to_canonical('fbfls') == 'flsss'

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
  logits = image_to_logits(image)
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
  if n_len == 1:
    return actions[1]
  else:
    return actions

def set_actions(files, directions, batch_no):
  actions = get_actions(files, batch_no)
  for action_no in range(n_len):
    directions[batch_no][action_to_one_hot_index(actions[action_no])] = 1

def get_image_sequence(data_dir, files, i):
  if get_counter(files, i) == 1:
    images = [load_image(data_dir, files[i+j]) for j in range(n_files)]
    return images

def index2action(index):
  return actions[index]

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

def tf_add_conv(inputs, filters, kernel_size=[5,5], include_pool=True):
  layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding='same', activation=tf.nn.relu)
  if include_pool:
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='same')
  return layer

model_input = tf.placeholder(tf.float32, shape=(None, n_files, n_rows, n_cols), name='model_input')
model_batch_size = tf.placeholder(tf.int32, [], name='batch_size')
model_keep_prob = tf.Variable(0.50, dtype=tf.float32)

def Model3D(model_input):
  layer = tf.reshape(model_input, (-1, n_files, n_rows, n_cols, 1))
  layer = tf.layers.conv3d(inputs=layer, filters=nb_filters, kernel_size=n_files, padding='same', activation=tf.nn.relu)
  layer = tf.layers.conv3d(inputs=layer, filters=nb_filters, kernel_size=n_files, padding='same', activation=tf.nn.relu)
  layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[4,2,2], strides=2, padding='same')
  layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[4,4,4], strides=2, padding='same')
  layer = tf.nn.dropout(layer, keep_prob=model_keep_prob)
  layer = tf.layers.flatten(layer)
  n_width = 50
  n_transition = n_len * n_width
  layer = tf.layers.dense(layer, n_transition, activation=tf.nn.relu)

  rnn_inputs = tf.split(layer, n_len, axis=1)
  def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(cell_state_size)
  
  model_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(rnn_cell_depth)])
  #model_initial_state = model_rnn_cell.zero_state(model_batch_size, tf.float32) 
  model_initial_state = model_rnn_cell.zero_state(model_batch_size, tf.float32) 
  model_rnn_outputs, model_rnn_state = tf.nn.static_rnn(model_rnn_cell, rnn_inputs, model_initial_state)
  model_logits = [tf.layers.dense(rnn_output, len(actions)) for rnn_output in model_rnn_outputs]

  return model_logits

model_logits = Model3D(model_input)
#model_softmax = tf.nn.softmax(model_logits)
#model_predict = tf.argmax(model_softmax, axis=1)
model_output = tf.placeholder(tf.int32, shape=[None, n_len, n_actions], name="model_output")

def loss():
  outputs = tf.split(model_output, n_len, axis=1)
  losses = []
  for i in range(n_len):
    losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs[i], logits=model_logits[i])))
  loss = tf.reduce_mean(losses)
  return loss, losses

model_loss, model_losses = loss()

def predict():
  predicts = []
  for i in range(n_len):
    predicts.append(tf.argmax(tf.nn.softmax(model_logits[i]), axis=1))
  return predicts

model_predict = predict()

model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op = model_optimizer.minimize(model_loss)
#model_accuracy = tf.metrics.accuracy(model_output, model_predict)

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

  nb_epochs = args.epochs

  X_train, Y_train, X_validation, Y_validation = load_training_data()

  def get_accuracy(X, Y): 
    predictions = session.run(model_predict, { model_batch_size: X.shape[0], model_input: X, model_output: batch_of_actions2one_hots(Y), model_keep_prob: 1.0 })
    predictions = np.array(predictions).transpose()
    predictions = batch_indexes2actions(predictions)
    right = len([True for (label, prediction) in zip(Y, predictions) if label == prediction])
    accuracy = float(right) / float(len(predictions)) 
    return accuracy, predictions

  if args.train:
    nb_train_samples, img_channels, n_rows, n_cols = X_train.shape
    nb_validation_samples = X_validation.shape[0]

    print(nb_train_samples, 'train samples')
    print(nb_validation_samples, 'validation samples')

    for epoch in range(nb_epochs):
      n_batches = nb_train_samples // batch_size
      indexes = list(range(nb_train_samples))
      random.shuffle(indexes)

      def select(elements, indexes, start, end):
        return [elements[indexes[i]] for i in range(start, end)] 

      for batch_no in range(n_batches):
        start = batch_no * batch_size
        end = start + batch_size
        batch_input = select(X_train, indexes, start, end)
        batch_output = batch_of_actions2one_hots(select(Y_train, indexes, start, end))
        loss, _ = session.run([model_loss, model_train_op], { model_batch_size: batch_size, model_input: batch_input, model_output: batch_output })
        '''
        loss, _, predict_, logits_ = session.run([model_loss, model_train_op, model_predict, model_logits], { model_batch_size: batch_size, model_input: batch_input, model_output: batch_output })
        if epoch == 1000:
          predict_1 = predict_
          logits_1 = logits_
          logits_p_1 = (np.array([softmax(l) for l in logits_1[0]])*100).astype(int)
          loss_1 = loss
        if epoch == nb_epochs-1:
          predict_end = predict_
          logits_end = logits_
          logits_p_end = (np.array([softmax(l) for l in logits_[0]])*100).astype(int)
          loss_end = loss
        '''
        #print("Train Loss {0}".format(loss))
   
      accuracy, predictions = get_accuracy(X_validation, Y_validation)
      accuracy_train, predictions_train = get_accuracy(X_train[0:32], Y_train[0:32])
      print("Epoch {0} Validation Accuracy: {1} Train Accuracy {2} Loss {3}".format(epoch, accuracy, accuracy_train, loss))
      #print("      tests {0}\n      predictions: {1}".format(Y_validation, predictions))
      #print("      tests {0} predictions: {1}".format(Y_train, predictions))
     

  if args.test:
    X_test, Y_test = load_test_data()
    predictions = session.run(model_predict, { model_batch_size: X_test.shape[0], model_input: X_test, model_output: batch_of_actions2one_hots(Y_test), model_keep_prob: 1.0 })
    predictions = np.array(predictions).transpose()
    predictions = batch_indexes2actions(predictions)
    right = len([True for (label, prediction) in zip(Y_test, predictions) if label == prediction])
    accuracy = float(right) / float(len(predictions)) 
    print("Test/Predictions")
    for t, p in zip(Y_test, predictions):
      print("{0}\n{1}\n".format(t, p))
    #print("test/predictions: {0}".format(zip(Y_test, predictions)))
    print("Test Accuracy: {0}".format(accuracy))

  saver.save(session, model_filename)

