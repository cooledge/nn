import numpy as np
import string
import os
import time
import pdb
import cv2
import socket
import tensorflow as tf
import utils
import argparse

parser = argparse.ArgumentParser(description="Runs neural nets on images")
parser.add_argument("--move", dest='move', default=True, action='store_true')
parser.add_argument("--no-move", dest='move', action='store_false')
parser.add_argument("--target", dest='target', default=True, action='store_true')
parser.add_argument("--no-target", dest='target', action='store_false')

args = parser.parse_args()

moving_graph = tf.Graph()
with moving_graph.as_default():
  from moving.move import Predict as Move_Predict
target_graph = tf.Graph()
with target_graph.as_default():
  from target.target import Predict as Target_Predict

def fn2idx(fn):
  return int(fn.strip(string.ascii_letters+"_."))

current_dir = './current'

def to_path(fn):
  return "{0}/{1}".format(current_dir, fn)

class FakePredict:
  def run(self, image):
    return "Output {0}".format(image.shape)

consumers= []
if args.move:
  consumers.append(Move_Predict())
if args.target:
  consumers.append(Target_Predict())
#consumers = [Move_Predict(), Target_Predict()]

'''
def si(image):
  cv2.imshow("", image)
  cv2.waitKey(1000)
'''

command = "ssh pi@robotcar 'echo {0} > ~/code/nn/robotcar/send_to.txt'".format(socket.gethostname())
os.system(command)

while True:
  time.sleep(0.1)
  files = os.listdir('./current')
  idx2fn = {}

  for file in files:
    idx2fn[fn2idx(file)] = file

  idxs = list(idx2fn.keys())
  # skip the last file since it might be getting written to
  idxs.pop()
  idxs.sort()
  for idx in idxs:
    try:
      filename = idx2fn[idx]
      filepath = to_path(filename)
      #image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
      #pdb.set_trace()
      image = utils.load_image(current_dir, filename, 480, 640)
      display_image = np.array(image)
      os.remove(filepath)
      #print("Run against neural net {0}".format(filename))
      for consumer in consumers:
        result = consumer.run(image, display_image)
        if result is not None:
          print(result)
      #pdb.set_trace()
      #utils.si(image, title='Input Image', waitTime=5000)
      utils.si(display_image, title='Output Image', waitTime=1)
    except IOError:
      print("IOError: {0}".format(filename))
      break
    except cv2.error:
      print("cv2.error")
      break
    except ValueError:
      print("Value : {0}".format(filename))
      break
