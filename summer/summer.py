import tensorflow as tf
import pdb

# Input: 2 Numbers and Op (Add, Subtract, Multiply, Divide)
# Output: the result of applying Op

ADD = 0
SUBTRACT = 1
MULTIPLY = 2

def to_one_hot(n, v):
  oh = [0 for _ in range(n)]
  oh[v] = 1
  return oh

x = []
y = []
for i in range(-10, 10):
  for j in range(-10, 10):
    for op in range(4):
      pdb.set_trace()
      x += [to_one_hot(20, i+10) + to_one_hot(20, j+10) + to_one_hot(4, op)]
      if op == ADD:
        y += [to_one_hot(200, i+j+100)]
      elif op == SUBTRACT:
        y += [to_one_hot(200, i-j+100)]
      elif op == MULTIPLY:
        y += [to_one_hot(200, i*j+100)]

pdb.set_trace()
