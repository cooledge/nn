#coding:utf-8

from time import ctime
from robotcar import RobotCar
import binascii
from socket import *
import RPi.GPIO as GPIO
import time
import pdb

LED0, LED1, LED2 = [10, 9, 25]
ENA, ENB = [13, 20]
IN1, IN2, IN3, IN4 = [19, 16, 21, 26]
SER1, SER2, SER3, SER4, SER7, SER8 = [11, 8, 7, 5, 6, 12]

class RobotCar_Hardware(RobotCar):

  # scale the time so the inverse directions
  # go the same distance. I got this by putting
  # that car in a spot and doing fb or lr and 
  # making sure it went back to the start
  # position
  def compensation(self):
    # fblrs
    return [1.0, 1.0, 1.05, 1.0, 1.0]

  def __init__(self):
    self.cap = None

    GPIO.setmode(GPIO.BCM)

    GPIO.setwarnings(False)

    GPIO.setup(LED0,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(LED1,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(LED2,GPIO.OUT,initial=GPIO.HIGH)

    # motor
    GPIO.setup(ENA,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(ENB,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)

    GPIO.setup(SER1,GPIO.OUT)
    GPIO.setup(SER2,GPIO.OUT)
    GPIO.setup(SER3,GPIO.OUT)
    GPIO.setup(SER4,GPIO.OUT)
    GPIO.setup(SER7,GPIO.OUT) # Horizontal servo port servo7
    GPIO.setup(SER8,GPIO.OUT) # Vertical servo port servo8
    '''
    Servo7=GPIO.PWM(SER7,50) #50HZ  
    Servo7.start(0)  
    Servo8=GPIO.PWM(SER8,50) #50HZ  
    Servo8.start(0)  
    '''

  def forward(self):
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,True)
    GPIO.output(IN2,False)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
    
  def backward(self):
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,False)
    GPIO.output(IN2,True)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
    
  def left(self):
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,True)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
    
  def right(self):
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,False)
    GPIO.output(IN2,True)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
    
  def stop(self):
    GPIO.output(ENA,False)
    GPIO.output(ENB,False)
    GPIO.output(IN1,False)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False)

  #Servo angle drive function   
  def SetServo7Angle(self, angle_from_protocol):
    angle=hex(eval('0x'+angle_from_protocol))
    angle=int(angle,16)
    Servo7.ChangeDutyCycle(2.5 + 10 * angle / 180) #set horizontal servo rotation angle

  def SetServo8Angle(self, angle_from_protocol):
    angle=hex(eval('0x'+angle_from_protocol))
    angle=int(angle,16)
    Servo8.ChangeDutyCycle(2.5 + 10 * angle / 180) #Set vertical servo rotation angel
    time.sleep(0.01)

  def camera_on(self):
    if self.cap:
      return

    import cv2
    self.cap = cv2.VideoCapture(0)

  def camera_off(self):
    self.cap.release()
    self.cap = None

  def camera_get_frame(self):
    self.camera_on()
    return self.cap.read()

  def camera_ok(self):
    self.camera_on()
    return self.cap.isOpened()

  '''
  def cleanup():
    self.stop()
    cv2.destroyAllWindows()

  def stream_camera(self, dst, prefix):
    atexit.register(cleanup)

    # get the first frame before starting the car moving
    # since there is a setup lag
    self.get_frame()
    #do_action()

    images = []
    start = time.time()
    while(td.isOpened()):
      duration = time.time() - start
      if duration > args.time:
        break

      ret, frame = td.get_frame()

      if ret==True:
        images.append(frame)
        if args.show:
          cv2.imshow('frame',frame)
      else:
        break

    print("Writing {0} files".format(len(images)))
    counter = 0
    timestr = time.strftime("%Y%m%d%H%M%S")
    for image in images:
      filename = "{0}/{1}_{2}-{3}.jpg".format(data_dir, args.direction, timestr, counter)
      counter += 1
      cv2.imwrite(filename, image)
  '''


if __name__ == '__main__':
  pdb.set_trace()
  rc = RobotCar_Hardware()
  rc.move("lr", 0.1)
  rc.stop()
  '''
  pdb.set_trace()
  for i in range(40):
    rc.backward()
    time.sleep(0.01)
    rc.stop() 
    time.sleep(0.01)
  '''

