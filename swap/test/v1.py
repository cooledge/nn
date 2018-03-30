import numpy as np
import cv2
import face_recognition
import pdb
import math
  
cap = cv2.VideoCapture(0)

def rotate_point(origin, point, angle):
  """
  Rotate a point counterclockwise by a given angle around a given origin.

  The angle should be given in radians.
  """
  ox, oy = origin
  px, py = point

  qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
  qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
  return qx, qy

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    def face_locations(frame, rotate):  
      cols = frame.shape[0]
      rows = frame.shape[1]
      origin = (cols/2,rows/2)
      r = cv2.getRotationMatrix2D(origin,rotate,1)
      frame_rotated = cv2.warpAffine(frame, r, (cols,rows))

      # Resize frame of video to 1/4 size for faster face recognition processing
      small_frame = cv2.resize(frame_rotated, (0, 0), fx=0.25, fy=0.25)

      # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
      fl_frame = small_frame[:, :, ::-1]
      #cv2.imshow('frame',fl_frame)
      locations = face_recognition.face_locations(fl_frame) 
  
      u_locations = []
      for location in locations:
        top, left, bottom, right = location
        top, left = rotate_point(origin, (top, left), -rotate)
        bottom, right = rotate_point(origin, (bottom, right), -rotate)
        u_locations.append([int(top), int(left), int(bottom), int(right)])

      return u_locations

    def label_face(frame, label):
      def add_label(frame, location):
        top, left, bottom, right = location
        top *= 4
        left *= 4
        bottom *= 4
        right *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom), (right, bottom+35), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (right + 6, bottom + 35 - 6), font, 1.0, (255, 255, 255), 1)

      #rotate = np.array([0,15,30,45,60,75,90])
      rotate = np.array([0])
      rotate = [val for pair in zip(rotate, rotate*-1) for val in pair][1:]

      locations = []
      for r in rotate:
        locations = face_locations(frame, r)
        if len(locations) > 0:
          break
        
      for l in locations:
        add_label(frame, l)

    label_face(frame, "Greg")
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
