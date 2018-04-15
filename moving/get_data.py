import numpy as np
import cv2
import pdb
import time
import math
import base64
import socket
import argparse
from flask import Flask

if socket.gethostname() == 'robotcar':
  from robotcar import RobotCar
else:
  from robotcar_stub import RobotCar

'''
get_data -dir forward -time 1 -label forward
'''
parser = argparse.ArgumentParser(description="Get data from the camera and motion")
parser.add_argument("--direction", type=str, default='forward', choices=['forward', 'backward', 'left', 'right', 'stop'])
parser.add_argument("--time", type=int, default=2000)
parser.add_argument('--no-show', dest='feature', action='store_false')
parser.set_defaults(show=True)
args = parser.parse_args()

args.time = args.time / 1000.0

robotcar = RobotCar()
if args.direction == 'forward':
  robotcar.forward()
elif args.direction == 'backward':
  robotcar.backward()
elif args.direction == 'left':
  robotcar.left()
elif args.direction == 'right':
  robotcar.right()

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

start = time.time()  
timestr = time.strftime("%Y%m%d-%H%M%S")

counter = 0
while(td.isOpened()):
  duration = time.time() - start
  if duration > args.time:
    break

  counter += 1
  print("Counter {0}".format(counter))
  ret, frame = td.get_frame()

  if ret==True:
    filename = "{0}/{1}{2}-{3}.jpg".format(data_dir, args.direction, timestr, counter)
    cv2.imwrite(filename, frame)
    if args.show:
      cv2.imshow('frame',frame)
  else:
    break

td.release()
cv2.destroyAllWindows()

'''
filespec = "{0}/{1}{2}*.jpg".format(data_dir, prefix, timestr, counter)
command = "scp {0} dev@ugpu:~/code/nn/unrotate/{1}".format(filespec, data_dir)
os.system(command)
'''


robotcar.stop()


