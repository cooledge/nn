import cv2
import pdb
import random
import time
import os
import math
import sys
import base64
import socket
import argparse
import atexit
from flask import Flask
import params

parent_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(parent_dir)

def inverse(actions):
  map = {'f': 'b', 'b': 'f', 'l': 'r', 'r': 'l', 's':'s'}
  return ''.join([map[action] for action in actions])[::-1]

assert inverse('fblrs') == 'slrfb'

if socket.gethostname() == 'robotcar':
  from robotcar_hardware import RobotCar_Hardware as RobotCar
else:
  from robotcar_stub import RobotCar_Stub as RobotCar

'''
get_data -dir forward -time 1 -label forward
'''
parser = argparse.ArgumentParser(description="Get data from the camera and motion")
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--time", type=int, default=2000)
parser.add_argument('--show', dest='show', default=False, action='store_true')
parser.add_argument('--steps', type=int, default=0)
parser.add_argument('--no-show', dest='show', action='store_false')
parser.set_defaults(show=False)
args = parser.parse_args()

args.time = args.time / 1000.0

robotcar = RobotCar()

class TestDataLocal:

  def __init__(self):
    self.cap = cv2.VideoCapture(0)

  def isOpened(self):
    return self.cap.isOpened()

  def get_frame(self):
    return self.cap.read()

  def release(self):
    self.cap.release
    self.cap = None

td = TestDataLocal()
data_dir = './data'

import os
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

def cleanup():
  print("stopping")
  robotcar.stop()
  td.release()
  cv2.destroyAllWindows()

atexit.register(cleanup)

# get the first frame before starting the car moving
# since there is a setup lag
td.get_frame()

seq_len = params.N_ACTIONS
def get_actions():
  return ''.join([random.choice(params.ACTIONS) for _ in range(seq_len)])

def error_check(ret):
  return 0

images = []
start = time.time()  
counter = 0
# learn direction and how much time to move left or right
step_dir = 'f'
STEP_DURATION = 0.05
if td.isOpened():
  for sample_no in range(args.n_samples):
    ret, before_frame = td.get_frame()
    error_check(ret)

    if args.steps > 0:
      if step_dir == 'f':
        robotcar.move('f', STEP_DURATION)
      else:
        robotcar.move('b', STEP_DURATION)
      robotcar.stop()
      counter += 1
      if counter >= args.steps:
        counter = 0
        if step_dir == 'b':
          step_dir = 'f'
        else:
          step_dir = 'b'

    time.sleep(1.0)

    actions = get_actions()
    print(actions)
    robotcar.move(actions, params.ACTION_DURATION)
    robotcar.stop()
    time.sleep(1.0)
       
    ret, after_frame = td.get_frame()
    error_check(ret)

    # back to start
    robotcar.move(inverse(actions), params.ACTION_DURATION)
    robotcar.stop()
    time.sleep(1.0)

    if args.show:
      cv2.imshow('before', before_frame)
      cv2.imshow('after', after_frame)

    images.append((before_frame, after_frame, actions))

print("Writing {0} example".format(len(images)))
timestr = time.strftime("%Y%m%d%H%M%S")
file_counter = 0
for before, after, actions in images:
  def wf(counter, image):
    filename = "{0}/{1}_{2}{3}-{4}.jpg".format(data_dir, actions, timestr, file_counter, counter)
    cv2.imwrite(filename, image)
  wf('1', before)
  wf('2', after)
  file_counter += 1

'''
filespec = "{0}/{1}{2}*.jpg".format(data_dir, prefix, timestr, counter)
command = "scp {0} dev@ugpu:~/code/nn/unrotate/{1}".format(filespec, data_dir)
os.system(command)
'''




