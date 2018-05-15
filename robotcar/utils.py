from PIL import Image
import numpy as np

def filepath(data_dir, filename):
  return "{0}/{1}".format(data_dir, filename)

def load_image(data_dir, filename, n_rows, n_cols):
  try:
    image = Image.open(filepath(data_dir, filename)).resize((n_cols, n_rows))
    image = np.array(image)
    image = np.mean(image, -1)
    return image / 255.
  except IOError:
    return None


