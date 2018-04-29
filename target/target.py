import cv2
import numpy as np
import sys
import os
import time
import pdb
import tensorflow as tf

arser = argparse.ArgumentParser(description="Detect direction of motion")

parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--train", dest='train', default=True, action='store_true')
parser.add_argument("--no-train", dest='train', action='store_false')
parser.add_argument("--test", dest='test', default=True, action='store_true')
parser.add_argument("--no-test", dest='test', action='store_false')
parser.add_argument("--clean", dest='clean', default=False, action='store_true')
parser.add_argument("--test-predict", dest='test_predict', default=False, type=bool)

args = parser.parse_args()

data_dir = './data'
files = os.listdir(data_dir)

# shape (NImages, Cols, Rows)
images = []

n_radius = 200
n_cols = 640
n_rows = 480

# y = (radius, col, row)
def filename_to_y(filename):
  return [int(c) for c in filename.split('.')[0].split('_')]

# X_train, Y_train_radius, y_train_position, X_validation, Y_validation_radius, Y_validation_position = load_training_data()

model_input = tf.placeholder(tf.float32, shape=(None, n_cols, n_rows))

model_logits = tf.placeholder(tf.float32, shape=(None, n_radius+n_cols*n_rows))
model_logits_radius, model_logits_position  = tf.split(model_logits, [n_radius, n_cols*n_rows], 1)
model_logits_position = tf.reshape(model_logits_position, (-1, n_cols, n_rows, 1))

model_output_radius = tf.placeholder(tf.float32, shape=(None, n_radius))
model_output_position = tf.placeholder(tf.float32, shape=(None, n_cols, n_rows))

def maxpool(layer):
  return tf.layers.max_pooling2d(tf.reshape(layer, (-1, n_cols, n_rows, 1)), 3, 2)

model_loss = tf.losses.mean_squared_error(maxpool(model_output_position), maxpool(model_logits_position))
model_optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)
model_train_op = model_optimizer.minimize(model_loss)

pdb.set_trace()

session = tf.Session()
session.run(tf.global_variables_initializer())

if __name__ == "__main__":
  X_train, Y_train_radius, y_train_position, X_validation, Y_validation_radius, Y_validation_position = load_training_data()

  if args.train:
    nb_train_samples, img_rows, img_cols = X_train.shape
    nb_validation_samples = X_validation.shape[0]

    print(nb_train_samples, 'train samples')
    print(nb_validation_samples, 'validation samples')

    for epoch in range(nb_epochs):
      n_batches = nb_train_samples // batch_size

      for batch_no in range(n_batches):
        start = batch_no * batch_size
        end = start + batch_size
        batch_input = X_train[start:end]
        
        batch_output_radius = []
        for radius in Y_train_radius[start:end]
          one_hot = np.zeros((n_radius))
          one_hot[radius] = 1
          batch_output_radius.append(one_hot)

        batch_output_position = []
        for (col, row) in Y_train_position[start:end]:
          one_hot = np.zeros((n_cols, n_rows))
          one_hot[col][row] = 1
          batch_output_position.append(one_hot)

        placeholders = { 
          model_input: batch_input, 
          model_output_radius: batch_output_radius, 
          model_output_position: batch_output_position 
        }
        loss, _ = session.run([model_loss, model_train_op], placeholders)
        #print("Train Loss {0}".format(loss))
      '''
      labels, predictions = session.run([model_output, model_predict], { model_input: X_validation, model_output: cats2inds(Y_validation), model_keep_prob: 1.0 })
      right = len([True for (label, prediction) in zip(labels, predictions) if label == prediction])
      accuracy = float(right) / float(len(predictions))
      print("Epoch {0} Validation Accuracy: {1} Loss {2}".format(epoch, accuracy, loss))
      #  accuracy = session.run(model_accuracy, { model_input: X_train, model_output: cats2inds(Y_train) })
      #  print("Epoch {0} Batch Accuracy: {1}".format(epoch, accuracy))
      '''
