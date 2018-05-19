import cv2

def draw_arrow(image, direction):
  h, w = image.shape
  r = int(h/2)
  c = int(w/2)

  if direction == 'stop':
    pt1 = (c-10, r-10)
    pt2 = (c+10, r+10)
    cv2.rectangle(image, pt1, pt2, (255,255,255), -1)
    return

  if direction == 'right':
    pt1 = (w-50, r)
    pt2 = (w-10, r)
  elif direction == 'left':
    pt1 = (50, r)
    pt2 = (10, r)
  elif direction == 'forward':
    pt1 = (c, 50)
    pt2 = (c, 10)
  elif direction == 'backward':
    pt1 = (c, h-50)
    pt2 = (c, h-10)

  cv2.arrowedLine(image, pt1, pt2, (255,255,255), 2)

if __name__ == '__main__':

  image = cv2.imread('image_1.jpg', cv2.IMREAD_GRAYSCALE)

  draw_arrow(image, 'right')
  draw_arrow(image, 'left')
  draw_arrow(image, 'forward')
  draw_arrow(image, 'backward')
  draw_arrow(image, 'stop')
  cv2.imshow('Image with arrow', image)
  cv2.waitKey(0)
