import tensorflow as tf
import pdb

session = tf.Session()

pdb.set_trace()
c1 = tf.constant([[[1],[2],[3]], [[4],[5],[6]]])
paddings = tf.constant([[0,0], [0,1], [0,0]])
v1 = tf.pad(c1, paddings, "CONSTANT")
print(session.run(v1))

model_inputs = tf.placeholder(tf.float32, [1,5,1])
model_conv = tf.layers.conv1d(model_inputs, 1, 2, padding='same')
'conv1d/kernel:0'
model_kernel = session.graph.get_tensor_by_name('conv1d/kernel:0')
inputs = [[[i] for i in range(5)]]
session.run(tf.global_variables_initializer())
session.run(tf.assign(model_kernel, tf.constant([[[1]],[[1]]], dtype=tf.float32)))
feed = { model_inputs: inputs }

conv = session.run(model_conv, feed_dict=feed)
print("conv {0}".format(conv))

model_maxpool = tf.layers.max_pooling1d(model_conv, 2, 1, padding='same')
maxpool = session.run(model_maxpool, feed_dict=feed)
print("maxpool {0}".format(maxpool))

# layer(batch_size, len, channels)
# kernel_size is output size
def conv1d_inverse(layer, n_filters=1, kernel_size=2, strides=1):
  n_seq_len = int(layer.get_shape()[1])
  n_filters_in = int(layer.get_shape()[2])
  
  extra = int((n_seq_len + kernel_size - 1) / kernel_size)*kernel_size - n_seq_len
  layer = tf.pad(layer, tf.constant([[0,0], [0,extra], [0,0]]), 'CONSTANT')
  features = tf.split(layer, layer.get_shape()[2], axis=2)
  features = [tf.squeeze(f, axis=2) for f in features]

  kernel = tf.ones([1, kernel_size, n_filters])
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

model_upsample = conv1d_inverse(model_maxpool, 1, 2)
upsample = session.run(model_upsample, feed_dict=feed)
print("maxpool {0}".format(upsample))

pdb.set_trace()
pdb.set_trace()

