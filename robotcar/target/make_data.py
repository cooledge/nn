import os
import cv2
import numpy as np
import pdb

# generate test data for finding image

backgrounds_dir = './backgrounds/'
data_dir = './data/'
os.path.exists(data_dir) or os.makedirs(data_dir)

backgrounds = os.listdir(backgrounds_dir)

n_rows = 480
n_cols = 640

for radius in range(10, 200, 10):
  print("Radius: {0}".format(radius))
  for row in range(radius, n_rows-radius, 20):
    for col in range(radius, n_cols-radius, 20):
      for idx_bg, background in enumerate(backgrounds):
        img = cv2.imread(backgrounds_dir+background)
        cv2.circle(img, (col, row), radius, (255,255,255), -1)
        #cv2.imshow('', img)
        #cv2.waitKey(1000)
        filename = "{0}_{1}_{2}_{3}.jpg".format(radius, row, col, idx_bg)
        #print(filename)
        cv2.imwrite(data_dir+filename, img)
    break