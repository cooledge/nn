import cv2
import pdb

print(cv2.__version__)  # my version is 3.1.0
vidcap = cv2.VideoCapture('jre.mp4')
pdb.set_trace()
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = vidcap.get(cv2.CAP_PROP_FPS)
fourcc = int(vidcap.get(cv2.CAP_PROP_FOURCC))
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
vidwri = cv2.VideoWriter('output.mp4',fourcc, fps, (width,height))

index = 0
while True:
  success,frame = vidcap.read()
  if not success:
    break
  #cv2.imwrite("frame{:d}.jpg".format(index), frame)
  vidwri.write(frame)
  index += 1

vidwri.release()
vidcap.release()

print("success: {0}".format(success))
