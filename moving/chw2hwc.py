import tensorflow as tf
import pdb

# Convert batch of channel,width,height images to
# width,height,channel

def chw2hwc(chw):
  _, channels, height, width = chw.shape
  layer = tf.reshape(chw, [-1, channels, height*width])
  layer = tf.transpose(layer)
  hwc = tf.reshape(layer, [-1, height, width, channels])
  return hwc

if __name__ == '__main__':
  print("Doing tests")

  input1 = [ [
    [ [1,4], [7,10] ],
    [ [2,5], [8,11] ],
    [ [3,6], [9,12] ],
  ] ]

  session = tf.Session()

  model_input = tf.placeholder(tf.float32, shape=(1, 3, 2, 2))
  output1 = session.run(model_input, {model_input: input1})

  input2 = [ [
    [ 1,4,7,10 ],
    [ 2,5,8,11 ],
    [ 3,6,9,12 ],
  ] ]

  _, channels, height, width = model_input.shape

  layer = tf.reshape(model_input, [-1, channels, height*width])
  output2 = session.run(layer, {model_input: input1})

  input3 = [ [
         [ 1.,  2.,  3.],
         [ 4.,  5.,  6.],
         [ 7.,  8.,  9.],
         [10., 11., 12.]
    ] ]

  layer = tf.transpose(layer)
  output3 = session.run(layer, {model_input: input1})

  layer = tf.reshape(layer, [-1, height, width, channels])
  output4 = session.run(layer, {model_input: input1})

  def check(input, output):
    for b in range(1):
      for r in range(height):
        for c in range(width):
          assert input[0][0][r][c] == output[0][r][c][0]
          assert input[0][1][r][c] == output[0][r][c][1]
          assert input[0][2][r][c] == output[0][r][c][2]


  check(input1, output4)
  model_output = chw2hwc(model_input)
  output = session.run(model_output, {model_input: input1})
  check(input1, output)

