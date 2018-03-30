from __future__ import print_function
import numpy as np
import cv2
import pdb

cap = cv2.VideoCapture('jre.mp4')

width=1280
height=720
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (width,height))

counter = 0
while(cap.isOpened()):
  print("counter {0}\r".format(counter), end="")
  counter += 1
  ret, frame = cap.read()
  frame = cv2.resize(frame, (width,height))
  if ret==True:
      frame = cv2.flip(frame,0)

      # write the flipped frame
      out.write(frame)

      cv2.imshow('frame',frame)
      if counter > 200:
        break
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  else:
      break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
