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
 
 
class Model:

  def labels_to_one_hot(self, labels):
    labels = np.reshape(labels, (number_of_batches*batch_size*self._batch_len, 1))
    one_hot = np.zeros((number_of_batches*batch_size*self._batch_len, number_of_letters))
    for i in range(labels.shape[0]):
      one_hot[i][labels[i]] = 1
    return np.reshape(one_hot, (number_of_batches*batch_size, -1))
   
  def labels_to_one_hot_b(self, labels):
    labels = np.reshape(labels, (batch_size*self._batch_len, 1))
    one_hot = np.zeros((batch_size*self._batch_len, number_of_letters))
    for i in range(labels.shape[0]):
      one_hot[i][labels[i]] = 1
    return np.reshape(one_hot, (batch_size, -1))

  def __init__(self, batch_len):
    self._batch_len = batch_len
    self._scope = "scope_" + str(batch_len)

  def get_training_data(self, train_text): 
    self._number_of_batches = train_size // (self._batch_len * batch_size)
    train_text = train_text[:self._number_of_batches*self._batch_len*batch_size]

    self._train_ids = [char2id(ch) for ch in train_text]
    self._train_ids = np.array_split(self._train_ids, self._number_of_batches)
     
    self._train_ids = np.reshape(self._train_ids, (self._number_of_batches*batch_size, -1))
    self._label_ids = self._train_ids.copy()
    for i in range(self._label_ids.shape[0]):
      self._label_ids[i] = np.flipud(self._label_ids[i])
     
    if not np.array_equal(self._train_ids[0], np.flipud(self._label_ids[0])):
      pdb.set_trace()

  def build(self):
    with tf.variable_scope(self._scope): 
      cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)
      rnn_cell = tf.contrib.rnn.MultiRNNCell([cell] * rnn_cell_depth)
       
      self._model_input = tf.placeholder(tf.int32, shape=(batch_size, self._batch_len), name="inputs")
      self._target_placeholder = tf.placeholder(tf.int32, shape=(batch_size, self._batch_len), name="outputs")
       
      w = tf.get_variable("w", shape=(cell_state_size, number_of_letters))
      b = tf.get_variable("b", shape=(number_of_letters))
       
      model_embedding = tf.get_variable("embedding", (number_of_letters, embedding_size), tf.float32)
      rnn_input = tf.nn.embedding_lookup(model_embedding, self._model_input)
      rnn_input = [tf.squeeze(t) for t in tf.split(rnn_input, self._batch_len, 1)]
       
      model_zero_state = rnn_cell.zero_state(batch_size, tf.float32)
      model_outputs, model_state = tf.contrib.legacy_seq2seq.rnn_decoder(rnn_input, model_zero_state, rnn_cell)
       
      # map the decoder outputs to the vocab

      '''
      # model_outputs: list of 20 of (25, 128)
      model_outputs = tf.concat(model_outputs, 0)
      # model_outputs: (500, 128)
      '''

      # model_outputs: list of 20 of (25, 128)
      assert len(model_outputs) == self._batch_len
      assert model_outputs[0].get_shape() == (25, cell_state_size)
      model_outputs = tf.concat(model_outputs, 1)
      # model_outputs: 25, seq_len*128
      assert model_outputs.get_shape() == (25, self._batch_len*cell_state_size)
      model_outputs = tf.reshape(model_outputs, (-1, cell_state_size))
      assert model_outputs.get_shape() == (batch_size*self._batch_len, cell_state_size)

      # (500, 27)
      model_logits = tf.matmul(model_outputs, w) + b
      self._model_loss = tf.reduce_sum(tf.contrib.legacy_seq2seq.sequence_loss_by_example([model_logits], [tf.reshape(self._target_placeholder, [-1])], [tf.ones([batch_size*self._batch_len])], number_of_letters)) / batch_size / self._batch_len
      self._model_prediction = tf.nn.softmax(model_logits, -1, name="probs")
       
      model_optimizer = tf.train.AdamOptimizer(0.002)
      self._model_trainop = model_optimizer.minimize(self._model_loss)

  def train(self):
    session.run(tf.global_variables_initializer())

    for batch in range(self._number_of_batches):
      batch_start = batch*batch_size
      batch_end = batch_start + batch_size

      feed = { self._model_input: self._train_ids[batch_start:batch_end], self._target_placeholder: self._label_ids[batch_start:batch_end] }
      _, loss, pred = session.run([self._model_trainop, self._model_loss, self._model_prediction], feed)
      if batch % 100 == 0:
        print("Loss at step %d of %d is %1.3f" % (batch, self._number_of_batches, loss))
        print("Minibatch perplexity %d" % perplexity(pred, self.labels_to_one_hot_b(self._label_ids[batch_start:batch_end])))

        p = np.reshape(pred, (batch_size, self._batch_len, -1))
        l = np.reshape(self._label_ids[batch_start:batch_end], (batch_size, self._batch_len, -1))
        t = np.reshape(self._train_ids[batch_start:batch_end], (batch_size, self._batch_len, -1))

        for i in range(batch_size):
          pstring = one_hots_to_string(p[i])
          lstring = one_hots_to_string(l[i])
          tstring = ids_to_string(t[i])
          print("input: '%s' actual: '%s' predicted: '%s'" % (tstring, lstring, pstring))

  def predict(self, text):
    pred_len = self._batch_len // 2
    preds = []
    while len(text) > 0: 
      test_ids = np.zeros((batch_size, self._batch_len), dtype=np.int32)
      for i in range(len(text)):
        if i >= self._batch_len or i >= len(text):
          break
        test_ids[0][i] = char2id(text[i])
     
      feed = { self._model_input: test_ids }
      pred = session.run(self._model_prediction, feed)
      pred = np.reshape(pred, (batch_size, self._batch_len, -1))
      pred = np.split(pred[0], self._batch_len)
      preds = pred[len(pred)-pred_len:len(pred)] + preds
      text = text[pred_len:]
    return preds

  def test_predict(self):
    while True:
      text = input("Enter a string: ").lower().replace('\n\r', ' ')
      pred = self.predict(text)
      pdb.set_trace()
      pred = np.vstack(pred)
      output = one_hots_to_string(pred)
      print("Output is '%s'" % (output))

  def test(self):
    while True:
      text = input("Enter a string: ").lower().replace('\n\r', ' ')
      test_ids = np.zeros((batch_size, self._batch_len), dtype=np.int32)
      for i in range(len(text)):
        test_ids[0][i] = char2id(text[i])
     
      feed = { self._model_input: test_ids }
      pred = session.run(self._model_prediction, feed)
      pred = np.reshape(pred, (batch_size, self._batch_len, -1))
      output = one_hots_to_string(pred[0])
      print("Output is '%s'" % (output))
   
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

  def get_model(batch_len):
    model = Model(batch_len)
    model.get_training_data(train_text)
    model.build()
    model.train() 
    #model.test()
    return model

  m3 = get_model(3) 

  m3.test_predict()
