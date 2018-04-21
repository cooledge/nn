import numpy as np
import cv2
import pdb
import math
import base64
import argparse
import os

parser = argparse.ArgumentParser(description="Detect direction of motion")
#parser.add_argument("--copy_to", type=str, default='dev@dev-X555QA')
parser.add_argument("--copy_to", type=str, default='dev@ugpu')
args = parser.parse_args()

def si(image):
  cv2.imshow("", image)
  cv2.waitKey(1)

counter = 0

os.system("rm /tmp/image_*.jpg")

cap = cv2.VideoCapture(0)

while True:

  ret, frame = cap.read()

  if ret:
    print("Counter {0}".format(counter))
    counter += 1

    frame = cv2.resize(frame, (64,64))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filename = "image_{0}.jpg".format(counter)
    filepath = "/tmp/{0}".format(filename)
    cv2.imwrite(filepath, frame)

    command = "sshpass -p bobobo scp {0} {1}:~/code/nn/moving/current".format(filepath, args.copy_to)
    os.system(command)

cap.release()
cv2.destroyAllWindows()

