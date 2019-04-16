import tensorflow as tf
import numpy as np
import pdb

tf.enable_eager_execution()
a = tf.constant([
    1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4,
    5,5,5,5,5,5,5,5,5,
    6,6,6,6,6,6,6,6,6,
    7,7,7,7,7,7,7,7,7,
    8,8,8,8,8,8,8,8,8,
    9,9,9,9,9,9,9,9,9,
    10,10,10,10,10,10,10,10,10,
    11,11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,12,
    13,13,13,13,13,13,13,13,13,
    14,14,14,14,14,14,14,14,14,
    15,15,15,15,15,15,15,15,15,
    16,16,16,16,16,16,16,16,16,
    17,17,17,17,17,17,17,17,17,
    18,18,18,18,18,18,18,18,18,
])
print(a)

print(tf.reshape(a, [9, 18]))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Reshape((9*2*9, 1), input_shape=(9*2*9,)))
model.add(tf.keras.layers.Conv1D(1, 18, strides=18))
model.set_weights(np.array(model.get_weights()) / np.array(model.get_weights()))
print(model.get_weights())
print(model.output_shape)
pdb.set_trace()
