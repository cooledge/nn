import time

class RobotCar:

  # moves is a character sequence of fblrs
  # the car will move that at time units per
  def move(self, moves, tm=0.1):
    for move in moves:
      if move == 'f':
        self.forward()
      elif move == 'b':
        self.backward()
      elif move == 'l':
        self.left()
      elif move == 'r':
        self.right()
      elif move == 's':
        self.stop()
      time.sleep(tm)

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
