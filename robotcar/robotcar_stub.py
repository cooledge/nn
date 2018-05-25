from robotcar import RobotCar
import pdb

class RobotCar_Stub(RobotCar):

  def forward(self):
    print("Forward")
    
  def backward(self):
    print("Backward")
    
  def left(self):
    print("Left")
    
  def right(self):
    print("Right")
    
  def stop(self):
    print("Stop")


if __name__ == '__main__':
  rc = RobotCar_Stub()
  rc.move("fblrs")
