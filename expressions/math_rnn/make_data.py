# build the test data

import pdb
import itertools
import argparse

parser = argparse.ArgumentParser(description="Generate the training and test data")
parser.add_argument("--train", default="1,2,3", type=str, help="List of lengths to generate training data for")
parser.add_argument("--test", default="4", type=str, help="List of lengths to generate test data for")
args = parser.parse_args()
train_len = [ int(s) for s in args.train.split(',')]
test_len = [ int(s) for s in args.test.split(',')]

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

  #inputs = list(op_to_priority.keys())
  inputs = []

  for length in lengths:
    add_inputs(itertools.product(op_to_priority.keys(), repeat=length), inputs)

  for op in inputs:
    input.write("{0}\n".format(op))
    output.write("{0}\n".format(select_output(op)))

  output.close()
  input.close()

make_io("train", train_len)
make_io("test", test_len)
