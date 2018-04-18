import numpy as np
import string
import os
import time
import pdb

def fn2idx(fn):
  return int(fn.strip(string.ascii_letters+"_."))

current_dir = './current'

def to_path(fn):
  return "{0}/{1}".format(current_dir, fn)

while True:
  time.sleep(0.1)
  files = os.listdir('./current')
  idx2fn = {}

  for file in files:
    idx2fn[fn2idx(file)] = file

  idxs = idx2fn.keys()
  idxs.sort()
  for idx in idxs:
    try:
      filename = idx2fn[idx]
      filepath = to_path(filename)
      image = np.load(filepath)
      os.remove(filepath)
      print("Run against neural net {0}".format(filename))
    except IOError:
      0 # ignore pick up later
