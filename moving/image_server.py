import numpy as np
import cv2
import pdb
import math
import base64
import argparse

parser = argparse.ArgumentParser(description="Detect direction of motion")
parser.add_argument("--copy_to", type=str, default='dev@dev-X555QA')
args = parser.parse_args()

counter = 0

while True:
  pdb.set_trace()
  cap = cv2.VideoCapture(0)

  ret, frame = cap.read()

  if ret:
    print("Counter {0}".format(counter))
    counter += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filename = "image_{0}".format(counter)
    filepath = "/tmp/{0}".format(filename)
    np.save(filepath, gray)

    command = "sshpass -p bobobo scp ./{0} {1}:~/code/nn/moving/current".format(filepath, args.copy_to)
    os.system(command)

  cv2.destroyAllWindows()

