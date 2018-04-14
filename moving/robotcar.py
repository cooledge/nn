#coding:utf-8

from time import ctime
import binascii
from socket import *
import RPi.GPIO as GPIO
import time

class RobotCar:

  LED0, LED1, LED2 = [10, 9, 25]
  ENA, ENB = [13, 20]
  IN1, IN2, IN3, IN4 = [19, 16, 21, 26]
  SER1, SER2, SER3, SER4, SER7, SER8 = [11, 8, 7, 5, 6, 12]

  def __init__(self):
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

  def Forward(self):
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,True)
    GPIO.output(IN2,False)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
    
  def Backward(self):
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,False)
    GPIO.output(IN2,True)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
    
  def Left(self):
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,True)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
    
  def Right(self):
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,False)
    GPIO.output(IN2,True)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
    
  def Stop(self):
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

