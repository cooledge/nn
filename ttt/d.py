import tensorflow as tf
import numpy as np
import pdb

tf.enable_eager_execution()

input = [
  [
    [1,2,3,4,5,6,7,8,9],
    [11,12,13,14,15,16,17,18,19],
    [21,22,23,24,25,26,27,28,29],
    [31,32,33,34,35,36,37,38,39],
    [41,42,43,44,45,46,47,48,49],
    [51,52,53,54,55,56,57,58,59],
    [61,62,63,64,65,66,67,68,69],
    [71,72,73,74,75,76,77,78,79],
    [81,82,83,84,85,86,87,88,89],
  ],
  [
    [1,2,3,4,5,6,7,8,9],
    [11,12,13,14,15,16,17,18,19],
    [21,22,23,24,25,26,27,28,29],
    [31,32,33,34,35,36,37,38,39],
    [41,42,43,44,45,46,47,48,49],
    [51,52,53,54,55,56,57,58,59],
    [61,62,63,64,65,66,67,68,69],
    [71,72,73,74,75,76,77,78,79],
    [81,82,83,84,85,86,87,88,89],
  ]
]

print(tf.constant(input).shape)

n_samples = len(input)
n_boards = 9

# get rows 
boards = tf.split(input, n_boards, 1)
boards = [tf.squeeze(board, 1) for board in boards]
rows = [tf.split(board, 3, 1) for board in boards]

# get cols

samples = tf.split(input, n_samples, 0)
samples = [tf.squeeze(sample, 0) for sample in samples]
samples = tf.convert_to_tensor(samples)

# samples (?, 9, 9)

n_features = 2

def board_to_features(board):
  pdb.set_trace()
  #board = tf.squeeze(board)

  row1 = tf.convert_to_tensor([board[0], board[1], board[2]])
  row2 = tf.convert_to_tensor([board[3], board[4], board[5]])
  row3 = tf.convert_to_tensor([board[6], board[7], board[8]])

  col1 = tf.convert_to_tensor([board[0], board[3], board[6]])
  col2 = tf.convert_to_tensor([board[1], board[4], board[7]])
  col3 = tf.convert_to_tensor([board[2], board[5], board[8]])

  diag1 = tf.convert_to_tensor([board[0], board[4], board[8]])
  diag2 = tf.convert_to_tensor([board[2], board[4], board[6]])

# 3 squares by 1 feature

# (3, n_features)
  feature = [1 for _ in range(n_features)] 
  features = tf.constant([feature, feature, feature]) 

# set trace
  print(tf.linalg.matmul([row2], features))
  components = [row1, row2, row3, col1, col2, col3, diag1, diag2]
  feature_layer = [tf.linalg.matmul([component], features)[0] for component in components]
  feature_layer = tf.convert_to_tensor(feature_layer)
  feature_layer = tf.reshape(feature_layer, (len(components)*n_features,))
  print("return type shape {0}".format(feature_layer.shape))
  pdb.set_trace()
  feature_layer.dtype
  return feature_layer

# 9, 9
def sample_to_features(sample):
  pdb.set_trace()
  mapped = tf.map_fn(board_to_features, sample, dtype=(tf.int32,))
  pdb.set_trace()
  pdb.set_trace()
 
sample_to_features(samples[0]) 

def setup(samples):
  pdb.set_trace()
  tf.map_fn(sample_to_features, samples)
  samples_of_boards = [ tf.split(sample[0], 9, 0) for sample in tf.split(samples, samples.shape[0])]

  sample_output = []
  for sample in samples_of_boards:
    boards = []
    for board in sample:
      boards.append(board_to_features(board))
    sample_output.append(boards)

  sample_output = tf.convert_to_tensor(sample_output)
  print(sample_output)

setup(samples)

pdb.set_trace()
#cols = [tf.reshape(board, (n_samples, 3, 3)) for board in boards]
#cols = [tf.split(board, n_boards) for board in boards]
#cols = [tf.transpose(board, perm=(0,2,1)) for board in boards]

