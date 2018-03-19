import tensorflow as tf
import pdb

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
re = "[0-9][abc]*[0-9]+"

# create a neural net that recognized the specified pattern

class RegExp:
  def __init__(self, alphabet, regexp, length):
    self.alphabet = alphabet
    alength = len(alphabet)
    self.regexp = regexp
    self.slength = length

  def compile():
    model_input = tf.placeholder(tf.float32, (slength, alength))
    zero_to_nine_edges = [1 if ch "0123456789" else 0 for ch in alphabet]
    abc_edges = [1 if ch "abc" else 0 for ch in alphabet]
    model_zton = tf.constant(zero_to_nine_edges, dtype=tf.float32)
    model_output = tf.matmul(model_input, model_zton)

  def index_of_ch(ch):
    return self.alphabet.index(ch)

  def input_to_one_hot(ch):
    oh = [0 for _ in range(self.alength)]
    oh[index_of_ch[ch]] = 1 
    return oh

  def inputs_to_one_hot(inputs):
    return [input_to_one_hot(ch) for ch in inputs]

  def run(inputs):

sess = tf.Session()
sess.run(tf.global_variables_initializer())

re = RegExp(alphabet, re)
re.compile()
re.run("1a")

