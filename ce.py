import tensorflow as tf

input = [ [1,0,0] ]
logits = [ [1,0,0] ]
model_input = tf.placeholder(tf.float32, shape=(1, 3))
model_logits = tf.placeholder(tffloat32, shape=(1, 3))
sess = tf.Session()
result = sess.run(tf.softmax_cross_entropy_with_logits(labels=labels

