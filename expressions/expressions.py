import tensorflow as tf
import numpy as np
import pdb

priorities = [
  ('+', '*'),
  ('-', '*'),
  ('+', '/'),
  ('-', '/'),
  ('*', '**'),
  ('/', '**'),
  ('**', '0'),
  ('c', '+'),
  ('c', '-'),
  ('0', 'c')
]


ops = ['c', '+', '-', '*', '/', '**', '0']

'''

input = [1,0,0,0], [0,1,0,0], [0,0,1,0]

def predict(input):
  input 
    [1,0,0,0] -> [0,0,0,0]
    [0,1,0,0] -> [0,0,0,0]
    [0,0,1,0] -> [1,1,0,0]
    [0,0,0,1] -> [1,1,0,0]
  
sum input
predict children
subtract from each input 
'''

id_to_op = {}
op_to_id = {}
for op in ops:
  id = len(id_to_op)
  id_to_op[id] = op
  op_to_id[op] = id

def from_op_to_id(op):
  if op in op_to_id.keys():
    return(op_to_id[op])
  else:
    return(op_to_id['c'])
 
number_of_ops = len(ops)
batch_size = len(priorities)
inputs = np.zeros((batch_size, number_of_ops))
labels = np.zeros((batch_size, number_of_ops))
batch_no = -1
for k, v in priorities:
  batch_no = batch_no + 1
  labels[batch_no][from_op_to_id(k)] = 1.0
  inputs[batch_no][from_op_to_id(v)] = 1.0

model_inputs = tf.placeholder(tf.float32, shape=(None, number_of_ops), name="inputs")
model_labels = tf.placeholder(tf.float32, shape=(None, number_of_ops), name="labels")
model_w = tf.get_variable("w", shape=(number_of_ops, number_of_ops), dtype=tf.float32)
model_b = tf.get_variable("b", shape=(number_of_ops), dtype=tf.float32)
model_logits =  tf.matmul(model_inputs, model_w) + model_b
model_loss = tf.losses.softmax_cross_entropy(model_labels, model_logits)
model_predict = tf.nn.softmax(model_logits)
model_optimizer = tf.train.AdamOptimizer(0.05)
model_train = model_optimizer.minimize(model_loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

epochs = 50
for epoch in range(epochs):
  feed_dict = { model_inputs: inputs, model_labels: labels }
  _, loss = session.run([model_train, model_loss], feed_dict)
  print("Epoch {0} loss {1}".format(epoch, loss))

# check it

ops = ['c', '+', '-', '*', '/', '**']
for op in ops:
  inputs = np.zeros((1, number_of_ops))
  inputs[0][from_op_to_id(op)] = 1
  predict, = session.run([model_predict], feed_dict = {model_inputs: inputs})
  predict = predict[0]
  found = []
  for i in range(len(predict)):
    if predict[i] > 0.25:
      found.append("{0} - {1:0.2f}".format(id_to_op[i], predict[i]))

  print("op {0} found {1}".format(op, found))

number_of_inputs = 20
number_of_outputs = 20

'''
   c+-*/(**)
   
   input:        000100    001000

   predict:      111000    100000

   combined:     211000 (sum across predict)

   reduced:      -2-1-1000 -2-10100
'''

model_combined = tf.squeeze(tf.reshape(tf.reduce_sum(tf.split(model_predict, number_of_ops, axis=1), 1), (1, -1)))
model_reduced = tf.nn.relu(model_inputs - model_combined)
model_op_idx = tf.arg_max(tf.reduce_sum(model_reduced, 1), 0)

def get_op(expression):
  inputs = np.zeros((len(expression), number_of_ops))
  for i in range(len(expression)):
    inputs[i][from_op_to_id(expression[i])] = 1
  reduced, combined, op_idx = session.run([model_reduced, model_combined, model_op_idx], feed_dict = { model_inputs: inputs })

  ''' 
  # debug prints 
  print("combined")
  print(combined)
  print("inputs")
  print(inputs)
  print("reduced")
  print(reduced)
  # end debug prints
  '''

  return op_idx

while True:
  ex_string = input("Enter an arithmetic expression: ")
  if ex_string == "":
    break

  expression = ex_string.split()
  print("Input Expression: {0}".format(expression))
  while len(expression) > 1:
    op_idx = get_op(expression)
    op = expression[op_idx]

    arg_l_idx = op_idx - 1
    l = float(expression[arg_l_idx])
    arg_r_idx = op_idx + 1
    r = float(expression[arg_r_idx])

    if op == '+':
      result = l + r
    elif op == '-':
      result = l - r
    elif op == '*':
      result = l * r
    elif op == '/':
      result = l / r
    elif op == '**':
      result = l ** r

    expression = expression[:arg_l_idx] + [result] + expression[arg_r_idx+1:]

  print("Output Expression: {0}".format(expression))



