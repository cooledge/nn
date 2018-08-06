import tensorflow as tf
import pdb

# layer(batch_size, len, channels)
# n_filters: number of features to output
# kernel_size: how many nodes in source to combine
# strides: usual
def conv1d_inverse(layer, n_filters=1, kernel_size=2, strides=1, initializer=tf.contrib.layers.xavier_initializer()):
  n_seq_len = int(layer.get_shape()[1])
  n_filters_in = int(layer.get_shape()[2])
  
  extra = int((n_seq_len + kernel_size - 1) / kernel_size)*kernel_size - n_seq_len
  layer = tf.pad(layer, tf.constant([[0,0], [0,extra], [0,0]]), 'CONSTANT')
  features = tf.split(layer, layer.get_shape()[2], axis=2)
  features = [tf.squeeze(f, axis=2) for f in features]

  #initializer = tf.contrib.layers.xavier_initializer()
  #initializer = tf.constant_initializer(1.0)
  #kernel = tf.ones([1, kernel_size, n_filters])
  kernel = initializer([1, kernel_size, n_filters])
  kernels = tf.split(kernel, n_filters, axis=2)
  kernels = [tf.squeeze(kernel, axis=2) for kernel in kernels]
 
  # feature(batch_size, seq_len)
  def apply_filter(feature, kernel):
   splits = tf.split(feature, feature.get_shape()[1], axis=1)[0:n_seq_len]
   outputs = [tf.matmul(split, kernel) for split in splits]
   output = tf.concat(outputs, axis=1)
   return output

  outputs = [apply_filter(feature, kernel) for feature, kernel in zip(features, kernels)]
  outputs = [tf.reshape(output, (-1, n_seq_len*kernel_size, 1)) for output in outputs]
  output = tf.concat(outputs, axis=2)
  return output

if __name__ == '__main__':
  session = tf.Session()

  '''
  pdb.set_trace()
  c1 = tf.constant([[[1,1],[2,2],[3,3]], [[4,4],[5,5],[6,6]]])
  paddings = tf.constant([[0,0], [0,1], [0,0]])
  v1 = tf.pad(c1, paddings, "CONSTANT")
  print(session.run(v1))
  '''

  def test1():
    n_filters = 2
    model_inputs = tf.placeholder(tf.float32, [1,5,n_filters])
    model_conv = tf.layers.conv1d(model_inputs, n_filters, 2, padding='same')
    'conv1d/kernel:0'
    model_kernel = session.graph.get_tensor_by_name('conv1d/kernel:0')
    inputs = [[([i]*n_filters) for i in range(5)]]
    session.run(tf.global_variables_initializer())
    kernel_init = tf.constant([[[1]*n_filters]*n_filters,[[1]*n_filters]*n_filters], dtype=tf.float32)
    session.run(tf.assign(model_kernel, kernel_init))
    feed = { model_inputs: inputs }

    conv = session.run(model_conv, feed_dict=feed)
    print("conv {0}".format(conv))

    model_maxpool = tf.layers.max_pooling1d(model_conv, 2, 1, padding='same')
    maxpool = session.run(model_maxpool, feed_dict=feed)
    print("maxpool {0}".format(maxpool))

    model_upsample = conv1d_inverse(model_maxpool, 2, 2, initializer=tf.constant_initializer(1.0))
    upsample = session.run(model_upsample, feed_dict=feed)
    print("upsample {0}".format(upsample))

    pdb.set_trace()
    pdb.set_trace()

  n_filters = 1
  model_inputs = tf.placeholder(tf.float32, [1,5,n_filters])
  model_conv = tf.layers.conv1d(model_inputs, n_filters, 2, padding='same', dilation_rate=4)
  'conv1d/kernel:0'
  model_kernel = session.graph.get_tensor_by_name('conv1d/kernel:0')
  inputs = [[([i]*n_filters) for i in range(5)]]
  session.run(tf.global_variables_initializer())
  kernel_init = tf.constant([[[1]*n_filters]*n_filters,[[1]*n_filters]*n_filters], dtype=tf.float32)
  session.run(tf.assign(model_kernel, kernel_init))
  feed = { model_inputs: inputs }

  print("inputs {0}".format(inputs))

  conv = session.run(model_conv, feed_dict=feed)
  print("conv {0}".format(conv))

  model_maxpool = tf.layers.max_pooling1d(model_conv, 2, 1, padding='same')
  maxpool = session.run(model_maxpool, feed_dict=feed)
  print("maxpool {0}".format(maxpool))

  model_upsample = conv1d_inverse(model_maxpool, 2, 2, initializer=tf.constant_initializer(1.0))
  upsample = session.run(model_upsample, feed_dict=feed)
  print("upsample {0}".format(upsample))

  pdb.set_trace()
  pdb.set_trace()

