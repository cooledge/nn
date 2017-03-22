import numpy as np
import tensorflow as tf
import pdb

ops = ['0', 'c', '+', '-','*','/']
number_of_ops = len(ops)
types = ['0','c','infix','prefix','postfix']
number_of_types = len(types)
max_expression_length = 50
token_length = max_expression_length + number_of_types + number_of_ops
max_stack = 2

start_position = 0
start_type = max_expression_length
start_op = start_type + number_of_types

op_code_to_index = {'0': 0, 'c': 1, '+': 2, '-': 3, '*': 4, '/': 5}
type_code_to_index = { "0": 0, "c": 1, "infix": 2, "prefix": 3, "postfix": 4 }

def set_token(token, position, type, op):
  token[position] = 1.0
  token[max_expression_length+type_code_to_index[type]] = 1.0
  token[max_expression_length+number_of_types+op_code_to_index[op]] = 1.0

inputs = np.zeros((max_expression_length, token_length))
# c + c * c
set_token(inputs[0], 0, 'c', 'c')
set_token(inputs[1], 1, 'infix', '+')
set_token(inputs[2], 2, 'c', 'c')
set_token(inputs[3], 3, 'infix', '*')
set_token(inputs[4], 4, 'c', 'c')

selector_stack = np.zeros((max_stack, max_expression_length))
selector_stack[0][3] = 1

expression = tf.placeholder(tf.float32, shape=(max_expression_length, token_length), name="expression")
model_selector_stack = tf.placeholder(tf.float32, shape=(max_stack, max_expression_length), name="selector_stack")

def get_type_one_hot(type_code, selector):
  return tf.one_hot([type_code_to_index[type_code]], number_of_types, on_value = selector * 1, dtype=tf.float32)[0]

def get_op_one_hot(op_code, selector):
  return tf.one_hot([1], number_of_ops, on_value = selector * 1, dtype=tf.float32)[0]

def token_components(token):
  return tf.split(token, [max_expression_length, number_of_types, number_of_ops])
  
# expression list of (2)  [0] is position [1] is type
# selector is in [0,1]
def infix(expression, position, selector):
  if position == 0 or position == len(expression)-1:
    return 0

  left = expression[position-1]
  right = expression[position+1]
  op = expression[position] 

  left_position, left_type, left_op = token_components(left)
  right_position, right_type, right_op = token_components(right)
  op_position, op_type, op_op = token_components(op)

  # set the out position
  position = selector * (left_position + right_position + op_position)
  type = get_type_one_hot('c', selector)
  op = get_op_one_hot('c', selector)

  return tf.concat([position, type, op], 0)

def copy(expression, position, selector):
  return(selector * expression[position])
   
expression_stack = [] 

for i in range(max_stack):
  expression_stack.append([])
  if i == 0:
    for position in range(max_expression_length):
      expression_stack[i].append( expression[position] )
  else:
    for position in range(max_expression_length):
      selector = model_selector_stack[i-1][position]
      value = infix(expression_stack[i-1], position, selector) + \
              copy(expression_stack[i-1], position, -1*(selector-1))
      expression_stack[i].append( value )

session = tf.Session()
session.run(tf.global_variables_initializer())
feed_dict = { expression: inputs, model_selector_stack: selector_stack }

es = session.run(expression_stack, feed_dict)

def one_hot_to_set(one_hot):
  set = []
  for i in range(len(one_hot)):
    if one_hot[i]:
      set.append(i)
  return set

def print_expression(expression):
  for token in expression:
    positions = one_hot_to_set(token[:start_type])
    types = one_hot_to_set(token[start_type:start_op])
    ops = one_hot_to_set(token[start_op:])

    if not positions == []:
      print("{0} types({1}) ops({2})".format(positions, types, ops))

pdb.set_trace()
for i in range(max_stack):
  print("Expression {0}".format(i))
  print_expression(es[i])





