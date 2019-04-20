import tensorflow as tf
import numpy as np
import pdb

tf.enable_eager_execution()
x = [
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
]
a = tf.constant(x)
print(a)

print(tf.reshape(a, [9, 18]))

y = [
  [1,1], [1,1], [2,2], [2,2], [3,3], [3,3]
]
print(tf.reshape(y, (12,)))

pdb.set_trace()
model = tf.keras.Sequential()
VOCAB_SIZE = 19
EMBEDDING_SIZE = 4
model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_shape=(9*2*9,)))
print(model.predict(tf.cast([[x]], tf.float32)))
model.add(tf.keras.layers.Reshape((9*18, EMBEDDING_SIZE)))
#pdb.set_trace()
#print(model.predict(tf.cast([[x]], tf.float32)))
model.add(tf.keras.layers.Conv1D(1, (18,), strides=(18,)))
print(model.predict(tf.cast([[x]], tf.float32)))

'''
weights = model.get_weights()[0]
bias = model.get_weights()[1]
weights = weights/weights
bias = [1.0]
model.set_weights([weights, bias])
'''


print(model.get_weights())
print(model.output_shape)
print(model.predict(tf.cast([[x]], tf.float32)))

pdb.set_trace()
print(model.predict(tf.cast([[x]], tf.float32)))
