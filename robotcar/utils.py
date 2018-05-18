from PIL import Image
import cv2
import numpy as np
import pdb

def si(image):
  cv2.imshow("", image)
  cv2.waitKey(1)

def filepath(data_dir, filename):
  return "{0}/{1}".format(data_dir, filename)

def load_image(data_dir, filename, n_rows, n_cols):
  try:
    #pdb.set_trace()
    image = Image.open(filepath(data_dir, filename)).resize((n_cols, n_rows))
    #image = cv2.imread(filepath(data_dir, filename), cv2.IMREAD_GRAYSCALE)
    #image = cv2.resize(image, (n_cols, n_rows))
    image = np.array(image)
    if image.shape == (n_rows, n_cols, 3):
      image = np.mean(image, -1)
    return image / 255.
  except IOError:
    return None

def translate_point_helper(old_shape, new_shape, old_point):
 scale = np.array(new_shape) / np.array(old_shape)
 return tuple((old_point * scale).astype(int))

def translate_point(old_image, new_image, old_point):
  translate_point_helper(old_image.shape, old_image.shape, old_point)

if __name__ == '__main__':
  pdb.set_trace()
  new_point = translate_point_helper( (50, 50), (25, 25), (12, 12) )

  print(new_point)
