# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import pdb
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
#url = 'http://mattmahoney.net/dc/'
url = 'http://www.gutenberg.org/dirs/2/7/0/2701/'

primes =  [1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069
, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151
, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223
, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291
, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373
, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451
, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511
, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583
, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657
, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733
, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811
, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889
, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987
, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053
, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129]

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  else:
    statinfo = os.stat(filename)
    print("Using existing file of size %d" % (statinfo.st_size))
  '''
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  '''
  return filename

#filename = maybe_download('text8.zip', 31344016)
filename = maybe_download('2701.zip', 31344016)


def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()

text = read_data(filename)
print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
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

embedding_size = 128
batch_size=64
num_unrollings=10

prime_embedding = np.zeros((vocabulary_size, embedding_size), dtype=np.float32)
for i in range(vocabulary_size):
  prime = primes[i]
  iprime = 1.0 /prime
  for j in range(embedding_size):
    prime_embedding[i][j] = iprime

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch, self._last_batch_one_hot = self._next_batch()

  def reset():
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch, self._last_batch_one_hot = self._next_batch()

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, 1), dtype=np.int32)
    batch_one_hot = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.int32)
    for b in range(self._batch_size):
      batch[b, 0] = char2id(self._text[self._cursor[b]])
      batch_one_hot[b, char2id(self._text[self._cursor[b]])] = 1.0
      if np.argmax(batch_one_hot[b]) != batch[b, 0]:
        pdb.set_trace()
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch, batch_one_hot

  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    batches_one_hot = [self._last_batch_one_hot]
    for step in range(self._num_unrollings):
      batch, batch_one_hot = self._next_batch()
      batches.append(batch)
      batches_one_hot.append(batch_one_hot)
    self._last_batch = batches[-1]
    self._last_batch_one_hot = batches_one_hot[-1]
    return batches, batches_one_hot

def characters_one_hot(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def characters(chars):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in chars]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()[0]))
print(batches2string(train_batches.next()[0]))
print(batches2string(valid_batches.next()[0]))
print(batches2string(valid_batches.next()[0]))

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution, include_prob=False):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  selection = None
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      selection = i
      break

  if selection == None:
    selection = len(distribution) - 1

  if include_prob:
    return [selection, distribution[selection]]
  else:
    return selection

'''
def sample_distribution_n(distribution, n):
  selections = []
  for _ in range(n):
    selection, prob = sample_distribution(distribution, n)
    selections.append( (selection, prob) )
  return selections
'''

def sample_distribution_n(distribution, n):
  d = dict()
  for i in range(len(distribution)):
    d[i] = distribution[i]

  selections = []
  sorted_keys = sorted(d, key=d.get, reverse=True)
  for r in sorted_keys:
    if len(selections) > n:
      break
    selections.append( (r, d[r]) )

  return selections

def sample_one_hot(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def sample_char(prediction):
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  return [ [ sample_distribution(prediction[0]) ] ]

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]

num_nodes = 128

graph = tf.Graph()
with graph.as_default():

  # Parameters:
  # Input gate: input, previous output, and bias.
  ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.
  cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  #w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.5, 0.5))
  b = tf.Variable(tf.zeros([vocabulary_size]))

  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return output_gate * tf.tanh(state), state

  #embedding = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], -1.0, 1.0))
  #embedding = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], -1.0, 1.0))
  #embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  embedding = tf.Variable(tf.constant(prime_embedding))

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.int32, shape=[batch_size,1]))
  train_inputs = train_data[:num_unrollings]

  train_data_one_hot = list()
  for _ in range(num_unrollings + 1):
    train_data_one_hot.append(tf.placeholder(tf.int32, shape=[batch_size,vocabulary_size]))
  train_labels_one_hot = train_data_one_hot[1:]  # labels are inputs shifted by one time step.

  # (batch_size, 1, embedding_size)
  train_inputs = [tf.squeeze(tf.nn.embedding_lookup(embedding, td)) for td in train_inputs]

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  #with tf.control_dependencies([saved_output.assign(output),
  #                              saved_state.assign(state)]):
    # Classifier.
    #logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
  logits = tf.matmul(tf.concat(outputs, axis=0), w) + b
  softmax = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.concat(train_labels_one_hot, axis=0))
  loss = tf.reduce_mean(softmax)

  # Optimizer.
  global_step = tf.Variable(0)
  #learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  learning_rate = tf.constant(0.010, dtype=tf.float32)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)

  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.int32, shape=[1, 1], name="sample_input")
  sample_input_embedded = tf.nn.embedding_lookup(embedding, sample_input)
  sample_input_embedded = tf.reshape(sample_input_embedded, (1, embedding_size))
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(sample_input_embedded, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

num_steps = 7001
summary_frequency = 700

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches, batches_one_hot = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    for i in range(num_unrollings + 1):
      feed_dict[train_data_one_hot[i]] = batches_one_hot[i]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0 and step > 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches_one_hot)[1:])
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        beam_size = 2
        for _ in range(5):
          feed = sample_char(random_distribution())
          sentence = id2char(feed[0][0])
          reset_sample_state.run()

          def get_prob(id, prob, beam_size):
            if beam_size == 0:
              return prob

            max_prob = 0.0
            for _ in range(beam_size):
              feed[0][0] = id
              prediction = sample_prediction.eval({sample_input: feed})
              selections = sample_distribution_n(prediction[0], beam_size)
              for (next_id, next_prob) in selections:
                max_prob = max([max_prob, get_prob(next_id, next_prob, beam_size-1)*prob])

            return max_prob
         
          for _ in range(79):
            prediction, prev_sample_state = session.run([sample_prediction, sample_state], {sample_input: feed})
            selected_id = sample_distribution(prediction[0])
            '''
            selections = sample_distribution(prediction[0], beam_size)
            selected_id = None
            selected_prob = 0
            for (id, prob) in selections:
              if get_prob(id, prob, beam_size) > selected_prob:
                selected_id = id 
                selected_prob = prob
            '''
            
            feed[0][0] = selected_id
            sentence += id2char(selected_id)
          print(sentence)
        print('=' * 80)
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        batches, batches_one_hot = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: batches[0]})
        valid_logprob = valid_logprob + logprob(predictions, batches_one_hot[1])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

