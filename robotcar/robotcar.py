import time
import pdb

class RobotCar:

  def compensation(self):
    return [1.0, 1.0, 1.0, 1.0, 1.0]

  # moves is a character sequence of fblrs
  # the car will move that at time units per
  def move(self, moves, tm=0.1):
    for move in moves:
      if move == 'f':
        c = self.compensation()[0]
        self.forward()
      elif move == 'b':
        c = self.compensation()[1]
        self.backward()
      elif move == 'l':
        c = self.compensation()[2]
        self.left()
      elif move == 'r':
        c = self.compensation()[3]
        self.right()
      elif move == 's':
        c = self.compensation()[4]
        self.stop()
      time.sleep(tm*c)

  def camera_on(self):
    # on
    print("Camera on")

  def camera_off(self):
    # off
    print("Camera off")

  def camera_get_frame(self):
    print("Camera get frame")
    return []

  def camera_ok(self):
    print("Camera ok")
    return false
