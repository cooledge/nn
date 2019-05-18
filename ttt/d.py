import tensorflow as tf
import numpy as np
import pdb

tf.enable_eager_execution()

n_embedding = 2

r = n_embedding
input = [
  [
    [[1]*r,[2]*r,[3]*r,[4]*r,[5]*r,[6]*r,[7]*r,[8]*r,[9]*r],
    [[11]*r,[12]*r,[13]*r,[14]*r,[15]*r,[16]*r,[17]*r,[18]*r,[19]*r],
    [[21]*r,[22]*r,[23]*r,[24]*r,[25]*r,[26]*r,[27]*r,[28]*r,[29]*r],
    [[31]*r,[32]*r,[33]*r,[34]*r,[35]*r,[36]*r,[37]*r,[38]*r,[39]*r],
    [[41]*r,[42]*r,[43]*r,[44]*r,[45]*r,[46]*r,[47]*r,[48]*r,[49]*r],
    [[51]*r,[52]*r,[53]*r,[54]*r,[55]*r,[56]*r,[57]*r,[58]*r,[59]*r],
    [[61]*r,[62]*r,[63]*r,[64]*r,[65]*r,[66]*r,[67]*r,[68]*r,[69]*r],
    [[71]*r,[72]*r,[73]*r,[74]*r,[75]*r,[76]*r,[77]*r,[78]*r,[79]*r],
    [[81]*r,[82]*r,[83]*r,[84]*r,[85]*r,[86]*r,[87]*r,[88]*r,[89]*r],
  ],

  [
    [[1]*r,[2]*r,[3]*r,[4]*r,[5]*r,[6]*r,[7]*r,[8]*r,[9]*r],
    [[11]*r,[12]*r,[13]*r,[14]*r,[15]*r,[16]*r,[17]*r,[18]*r,[19]*r],
    [[21]*r,[22]*r,[23]*r,[24]*r,[25]*r,[26]*r,[27]*r,[28]*r,[29]*r],
    [[31]*r,[32]*r,[33]*r,[34]*r,[35]*r,[36]*r,[37]*r,[38]*r,[39]*r],
    [[41]*r,[42]*r,[43]*r,[44]*r,[45]*r,[46]*r,[47]*r,[48]*r,[49]*r],
    [[51]*r,[52]*r,[53]*r,[54]*r,[55]*r,[56]*r,[57]*r,[58]*r,[59]*r],
    [[61]*r,[62]*r,[63]*r,[64]*r,[65]*r,[66]*r,[67]*r,[68]*r,[69]*r],
    [[71]*r,[72]*r,[73]*r,[74]*r,[75]*r,[76]*r,[77]*r,[78]*r,[79]*r],
    [[81]*r,[82]*r,[83]*r,[84]*r,[85]*r,[86]*r,[87]*r,[88]*r,[89]*r],
  ],

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

n_features = 1

def board_to_features(board):
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
  features = tf.constant([feature]*3*n_embedding) 

  # set trace
  print(tf.linalg.matmul([tf.reshape(row2, (3*n_embedding,))], features))
  components = [row1, row2, row3, col1, col2, col3, diag1, diag2]
  feature_layer = [tf.linalg.matmul([tf.reshape(component, (3*n_embedding,))], features)[0] for component in components]
  feature_layer = tf.convert_to_tensor(feature_layer)
  feature_layer = tf.reshape(feature_layer, (len(components)*n_features,))

  return feature_layer

# 9, 9
def sample_to_features(sample):
  mapped = tf.map_fn(board_to_features, sample, dtype=tf.int32)
  return mapped

#pdb.set_trace() 
#sample_to_features(samples[0]) 
#pdb.set_trace() 

def setup(samples):
  #pdb.set_trace()
  output = tf.map_fn(sample_to_features, samples)
  print(output)

setup(samples)

print(samples)
pdb.set_trace()
#cols = [tf.reshape(board, (n_samples, 3, 3)) for board in boards]
#cols = [tf.split(board, n_boards) for board in boards]
#cols = [tf.transpose(board, perm=(0,2,1)) for board in boards]

