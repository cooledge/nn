import cv2
import numpy as np
import sys
import pdb
 
sys.path.insert(0, '../moving')

from robotcar_hardware import RobotCar_Hardware

rc = RobotCar_Hardware()

while True:
  print("Loop")
  (ok, cimg) = rc.camera_get_frame()

  if not ok:
    break

  img = cv2.medianBlur(cimg,5)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)

  circles = np.uint16(np.around(circles))
  for i in circles[0,:]:
      # draw the outer circle
      cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
      # draw the center of the circle
      cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3) 
  
  cv2.imshow('Image from RobotCar', cimg)
  #cv2.waitKey(0)

rc.camera_off()


