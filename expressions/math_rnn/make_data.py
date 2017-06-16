# build the test data

import pdb
import itertools

operators = "+-*/^"

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


def add_inputs(tuples, inputs):
  for t in tuples:
    ops = ""
    for op in t:
      ops += op
    inputs += [ops]

def make_io(suffix, lengths):
  input = open("input_{0}.txt".format(suffix), "w")
  output = open("output_{0}.txt".format(suffix), "w")

  inputs = list(op_to_priority.keys())

  for length in lengths:
    add_inputs(itertools.product(op_to_priority.keys(), repeat=length), inputs)

  for op in inputs:
    input.write("{0}\n".format(op))
    output.write("{0}\n".format(select_output(op)))

  output.close()
  input.close()

make_io("train", [2,3,4])
make_io("test", [5])
