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

