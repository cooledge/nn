# build the test data

import pdb
import itertools

operators = "+-*/^"

input = open('input.txt', "w")
output = open('output.txt', "w")

op_to_priority = { '+': 1, '-': 1, '*': 2, '/': 3, '^':4 }

ops = ""
for op in op_to_priority:
  ops += op

def select_output(input):
  priority = 0
  for op in input:
    if op_to_priority[op] > priority:
      sel_op = op
      priority = op_to_priority[op]
  return sel_op

inputs = list(op_to_priority.keys())

def add_inputs(tuples, inputs):
  for t in tuples:
    ops = ""
    for op in t:
      ops += op
    inputs += [ops]

add_inputs(itertools.permutations(op_to_priority.keys(), 2), inputs)
add_inputs(itertools.permutations(op_to_priority.keys(), 3), inputs)
add_inputs(itertools.permutations(op_to_priority.keys(), 4), inputs)

for op in inputs:
  input.write("{0}\n".format(op))
  output.write("{0}\n".format(select_output(op)))

output.close()
input.close()
