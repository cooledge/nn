import Tkinter as tk
from PIL import Image
from PIL import ImageTk
import numpy
import cv2
import pdb

class VideoIn:

  def on_snap_pressed(self):
    self.cap = cv2.VideoCapture(0)
    ret, frame = self.cap.read()
    # frame.shape = (480,640,3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_out = self.on_image(gray)
    self.show_image(image_out)
    cv2.waitKey(1)
    self.cap.release()

  def show_image(self, gray): 
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(image)
    image.thumbnail((gray.shape[0]/2,gray.shape[1]/2))

    ph = ImageTk.PhotoImage(image)

    if self.image is None:
      self.image = tk.Label(image=ph)
      self.image.image = ph
      self.image.pack(side=tk.RIGHT, padx=10, pady=10)
    else:
      self.image.image = ph
      self.image.configure(image=ph)
      self.image.pack(side=tk.RIGHT, padx=10, pady=10)

  def __init__(self):
    self.image = None

  def run(self):
    top = tk.Tk()

    action_button = tk.Button(top, text="Snap", command=(lambda: self.on_snap_pressed()))
    action_button.pack()

    self.image = tk.Label()
    self.image.pack(side=tk.RIGHT, padx=10, pady=10)

    self.text = tk.Text(top)
    self.text.insert(tk.INSERT, "This is where the text will appear")
    self.text.pack()

    top.mainloop()

    cv2.destroyAllWindows()

if __name__ == '__main__':
  class VI(VideoIn):
    #def __init__(self):

    def on_image(self, gray):
      image_out = gray
      return image_out

  vi = VI()
  vi.run()

