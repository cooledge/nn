import tensorflow as tf
import numpy as np

# neural net that learns to add two numbers between 1 and 100

max_numbers = 100

def int_to_one_hot(i, max):
  one_hot = np.zeros((max))
  one_hot[i] = 1
  return one_hot

def ints_to_one_hots(ints, max):
  return [int_to_one_hot(i, max) for i in ints]

def one_hot_to_int(one_hot):
  return np.argmax(one_hot)

model_input_number1 = tf.placeholder(tf.float32, shape=(?, max_numbers))
model_input_number2 = tf.placeholder(tf.float32, shape=(?, max_numbers))

model_input = f(model_input_number1, model_input_number2)

model_output = tf.placeholder(tf.float32, shape=(?, max_numbers*2))

batch_size = 32
n_samples = max_numbers * max_numbers

def make_samples():
  samples_number1 = []
  samples_number2 = []
  samples_total = []
  for i in range(max_numbers):
    for j in range(max_numbers):
      samples_number1.append(i)
      samples_number2.append(j)
      samples_total.append(i+j)
  return (samples_number1, samples_number2, samples_total)
   
samples_number1, samples_number2, samples_total  = make_samples()
   
session = tf.Session()
session.run(tf.global_variables_initializer())

epochs = 500
for epoch in range(epochs):
  n_batches = n_samples / batch_size
  for batch in range(n_batches):
    start = batch * batch_size
    end = start + batch_size
    ph = {
      model_input_number1: ints_to_one_hots(samples_number1[start:end], max_numbers),
      model_input_number2: ints_to_one_hots(samples_number2[start:end], max_numbers),
      model_output: ints_to_one_hots(samples_total[start:end], max_numbers*2)
    }
    loss, _ = tf.run([model_loss, model_train_op], ph)
    print("Loss {0}".format(loss)

  right = 0
  for i in range(max_numbers):
    for j in range(max_numbers):
      ph = {
        model_input_number1: ints_to_one_hots([i]),
        model_input_number2: ints_to_one_hots([j]),
        model_outputs: ints_to_one_hots([i+j])
      }
      predict_one_hot = tf.run(model_predict, ph)
      predict = np.argmax(predict_one_hot)
      if predict == i+j:
        right += 1.0

  print("Accuracy: {0}".format(right/(max_numbers*max_numbers)))
      

