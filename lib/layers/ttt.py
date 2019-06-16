import tensorflow as tf
from tensorflow import keras

class TTTLayer(tf.keras.layers.Layer):

  def __init__(self, n_embedding, n_features):
    super(TTTLayer, self).__init__()
    self.n_features = n_features
    self.n_embedding = n_embedding

  def get_config(self):
    config = {'n_embedding': self.n_embedding,
              'n_features': self.n_features,
             }
    base_config = super(TTTLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
    
  # (N, 9)
  def build(self, input_shape):
    assert input_shape[-1] == self.n_embedding
    assert input_shape[-2] == 9
    self.kernel = []
    self.kernel = self.add_variable("TTTLayer_kernel", shape=[3*self.n_embedding, self.n_features])

  def f(self, t):
    return tf.reshape(t, (1, self.n_embedding))

  def board_to_features(self, board):

    row1 = tf.concat([TTTLayer.f(self, board[0]), TTTLayer.f(self, board[1]), TTTLayer.f(self, board[2])], 0)
    row2 = tf.concat([TTTLayer.f(self, board[3]), TTTLayer.f(self, board[4]), TTTLayer.f(self, board[5])], 0)
    row3 = tf.concat([TTTLayer.f(self, board[6]), TTTLayer.f(self, board[7]), TTTLayer.f(self, board[8])], 0)

    col1 = tf.concat([TTTLayer.f(self, board[0]), TTTLayer.f(self, board[3]), TTTLayer.f(self, board[6])], 0)
    col2 = tf.concat([TTTLayer.f(self, board[1]), TTTLayer.f(self, board[4]), TTTLayer.f(self, board[7])], 0)
    col3 = tf.concat([TTTLayer.f(self, board[2]), TTTLayer.f(self, board[5]), TTTLayer.f(self, board[8])], 0)

    diag1 = tf.concat([TTTLayer.f(self, board[0]), TTTLayer.f(self, board[4]), TTTLayer.f(self, board[8])], 0)
    diag2 = tf.concat([TTTLayer.f(self, board[2]), TTTLayer.f(self, board[4]), TTTLayer.f(self, board[6])], 0)

    components = [row1, row2, row3, col1, col2, col3, diag1, diag2]
    feature_layer = [tf.linalg.matmul([tf.reshape(component, (3*self.n_embedding,))], self.kernel)[0] for component in components]
    feature_layer = tf.concat(feature_layer, 0)
    feature_layer = tf.reshape(feature_layer, (len(components)*self.n_features,))

    return feature_layer

  # 9, 9
  def sample_to_features(self, sample):
    mapped = tf.map_fn(lambda board: TTTLayer.board_to_features(self, board), sample, dtype=tf.float32)
    return mapped

  # input (?, Choices, BoaGamesrd)
  def call(self, samples):
    output = tf.map_fn(lambda sample: TTTLayer.sample_to_features(self, sample), samples)
    return output

