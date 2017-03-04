# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import pdb
 
url = 'http://mattmahoney.net/dc/'
#url = "http://www.gutenberg.org/dirs/2/7/0/2701/"
 
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    print("Getting file %s" % (url+filename))
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size > expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename
 
#filename = maybe_download('moby_dick.zip', 100000)
filename = maybe_download('2701.zip', 100000)
 
def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()
 
text = read_data(filename).lower().replace('\n\r', ' ')
print('Data size %d' % len(text))
print(text[0:100])
 
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])
 
number_of_letters = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])
 
def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    #print('Unexpected character: %s' % char)
    return 0
 
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '
 
print(char2id('a'), char2id('z'), char2id(' '))
print(id2char(1), id2char(26), id2char(0))
 
batch_size = 25
batch_len = 20
rnn_cell_depth = 1
embedding_size = 128
cell_state_size = 128
 
number_of_batches = train_size // (batch_len * batch_size)
train_text = train_text[:number_of_batches*batch_len*batch_size]
 
def labels_to_one_hot(labels):
  labels = np.reshape(labels, (number_of_batches*batch_size*batch_len, 1))
  one_hot = np.zeros((number_of_batches*batch_size*batch_len, number_of_letters))
  for i in range(labels.shape[0]):
    one_hot[i][labels[i]] = 1
  return np.reshape(one_hot, (number_of_batches*batch_size, -1))
 
def labels_to_one_hot_b(labels):
  labels = np.reshape(labels, (batch_size*batch_len, 1))
  one_hot = np.zeros((batch_size*batch_len, number_of_letters))
  for i in range(labels.shape[0]):
    one_hot[i][labels[i]] = 1
  return np.reshape(one_hot, (batch_size, -1))
 
train_ids = [char2id(ch) for ch in train_text]
train_ids = np.array_split(train_ids, number_of_batches)
 
train_ids = np.reshape(train_ids, (number_of_batches*batch_size, -1))
label_ids = train_ids.copy()
for i in range(label_ids.shape[0]):
  label_ids[i] = np.flipud(label_ids[i])
 
#check data
if not np.array_equal(train_ids[0], np.flipud(label_ids[0])):
  pdb.set_trace()
 
#label_ids = labels_to_one_hot(label_ids)
 
# build model
 
cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)
rnn_cell = tf.contrib.rnn.MultiRNNCell([cell] * rnn_cell_depth)
 
model_input = tf.placeholder(tf.int32, shape=(batch_size, batch_len), name="inputs")
target_placeholder = tf.placeholder(tf.int32, shape=(batch_size, batch_len), name="outputs")
 
w = tf.get_variable("w", shape=(cell_state_size, number_of_letters))
b = tf.get_variable("b", shape=(number_of_letters))
 
model_embedding = tf.get_variable("embedding", (number_of_letters, embedding_size), tf.float32)
rnn_input = tf.nn.embedding_lookup(model_embedding, model_input)
rnn_input = [tf.squeeze(t) for t in tf.split(rnn_input, batch_len, 1)]
 
model_zero_state = rnn_cell.zero_state(batch_size, tf.float32)
model_outputs, model_state = tf.contrib.legacy_seq2seq.rnn_decoder(rnn_input, model_zero_state, rnn_cell)
 
# map the decoder outputs to the vocab

'''
# model_outputs: list of 20 of (25, 128)
model_outputs = tf.concat(model_outputs, 0)
# model_outputs: (500, 128)
'''

# model_outputs: list of 20 of (25, 128)
assert len(model_outputs) == batch_len
assert model_outputs[0].get_shape() == (25, cell_state_size)
model_outputs = tf.concat(model_outputs, 1)
# model_outputs: 25, seq_len*128
assert model_outputs.get_shape() == (25, batch_len*cell_state_size)
model_outputs = tf.reshape(model_outputs, (-1, cell_state_size))
assert model_outputs.get_shape() == (batch_size*batch_len, cell_state_size)

# (500, 27)
model_logits = tf.matmul(model_outputs, w) + b
model_loss = tf.reduce_sum(tf.contrib.legacy_seq2seq.sequence_loss_by_example([model_logits], [tf.reshape(target_placeholder, [-1])], [tf.ones([batch_size*batch_len])], number_of_letters)) / batch_size / batch_len
model_prediction = tf.nn.softmax(model_logits, -1, name="probs")
 
model_optimizer = tf.train.AdamOptimizer(0.002)
model_trainop = model_optimizer.minimize(model_loss)
 
def logprob(predictions, labels):
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]
 
def perplexity(predictions, labels):
  labels = np.reshape(labels, (-1, number_of_letters))
  return(np.exp(logprob(predictions, labels)))
 
def one_hots_to_string(one_hots):
  output = ""
  for one_hot in one_hots:
    output = output + id2char(np.argmax(one_hot))
  return output
 
def ids_to_string(ids):
  output = ""
  for id in ids:
    output = output + id2char(id)
  return output
 
def print_one_hot_batch(batches):
  for i in range(batches.shape[0]):
    print(one_hots_to_string(batches[i]))
 
def print_batch(batches):
  for i in range(batches.shape[0]):
    print(ids_to_string(batches[i]))
 
# train
 
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
 
  for batch in range(number_of_batches):
    batch_start = batch*batch_size
    batch_end = batch_start + batch_size

    feed = { model_input: train_ids[batch_start:batch_end], target_placeholder: label_ids[batch_start:batch_end] }
    _, loss, pred = session.run([model_trainop, model_loss, model_prediction], feed)
    if batch % 100 == 0:
      print("Loss at step %d of %d is %1.3f" % (batch, number_of_batches, loss))
      print("Minibatch perplexity %d" % perplexity(pred, labels_to_one_hot_b(label_ids[batch_start:batch_end])))

      p = np.reshape(pred, (batch_size, batch_len, -1))
      l = np.reshape(label_ids[batch_start:batch_end], (batch_size, batch_len, -1))
      t = np.reshape(train_ids[batch_start:batch_end], (batch_size, batch_len, -1))

      for i in range(batch_size):
        pstring = one_hots_to_string(p[i])
        lstring = one_hots_to_string(l[i])
        tstring = ids_to_string(t[i])
        print("input: '%s' actual: '%s' predicted: '%s'" % (tstring, lstring, pstring))
 
  # test
 
  while True:
    text = input("Enter a string: ").lower().replace('\n\r', ' ')
    test_ids = np.zeros((batch_size, batch_len), dtype=np.int32)
    for i in range(len(text)):
      test_ids[0][i] = char2id(text[i])
   
    feed = { model_input: test_ids }
    pred = session.run(model_prediction, feed)
    pred = np.reshape(pred, (batch_size, batch_len, -1))
    output = one_hots_to_string(pred[0])
    print("Output is '%s'" % (output))


