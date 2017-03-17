# nn repeat the training data more for longer runs
# random average is 20 / model is 9 one time thought it got 100
import gym
import pdb
import tensorflow as tf
import numpy as np
import random

env = gym.make('CartPole-v0')

number_of_actions = env.action_space.n
number_of_observations = env.observation_space.shape[0]
number_of_inputs = number_of_observations

model_inputs = tf.placeholder(tf.float32, shape=(None, number_of_inputs), name="Inputs")
model_actions = tf.placeholder(tf.float32, shape=(None, number_of_actions), name="Actions")
model_w = tf.get_variable("w", shape=(number_of_inputs, number_of_actions), dtype=tf.float32)
model_b = tf.get_variable("b", shape=(number_of_actions), dtype=tf.float32)
model_logits = tf.matmul(model_inputs, model_w) + model_b
model_predict = tf.nn.softmax(model_logits)
model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_actions, logits=model_logits))
model_optimizer = tf.train.AdamOptimizer(0.01)
model_train_op = model_optimizer.minimize(model_loss)

class TrainingData:
  def __init__(self):
    self.training_data = []

  def add(self, reward, observations, actions):
    repeats = int(reward // 5 + 1)
    for observation, action in zip(observations*repeats, actions*repeats):
      self.training_data += [(observation, action)]

  def start(self, batch_size):
    self.batch_size = batch_size
    self.next_i = 0

  def done(self):
    return self.next_i + self.batch_size > len(self.training_data)

  def next_batch(self, inputs, labels):
    for i in range(self.batch_size):
      td = self.training_data[i+self.next_i]
      inputs[i] = td[0]
      labels[i][td[1]] = 1.0
    self.next_i += self.batch_size
  
  def shuffle(self):
    random.shuffle(self.training_data)
    
env.reset()
training_data = TrainingData()
number_of_games = 20
overall_total = 0
for i_episode in range(number_of_games):
  observation = env.reset()
  observations = [observation]
  actions = []
  total_reward = 0
  for t in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    actions += [action]
    total_reward += reward
    if done:
      training_data.add(total_reward, observations, actions)
      break
    observations += [observation]
  overall_total += total_reward
average_reward = overall_total / number_of_games
print("Average reward is {0}".format(average_reward))

session = tf.Session()
session.run(tf.global_variables_initializer())

training_data.shuffle()
batch_size = 20
training_data.start(batch_size)
while not training_data.done():
  inputs = np.zeros((batch_size, number_of_observations))
  labels = np.zeros((batch_size, number_of_actions))
  training_data.done()
  training_data.next_batch(inputs, labels)
  feed_dict = {model_inputs: inputs, model_actions: labels}
  loss, _ = session.run([model_loss, model_train_op], feed_dict = feed_dict)
  print("Loss is {0}".format(loss))

number_of_plays = 20

def play(number_of_games):
  env.reset()
  overall_total = 0
  for i_episode in range(number_of_games):
    observation = env.reset()
    for t in range(100):
      env.render()
      #action = env.action_space.sample()

      feed_dict = {model_inputs: [observation]}
      predict = session.run([model_predict], feed_dict = feed_dict)
      action = np.argmax(predict)

      observation, reward, done, info = env.step(action)
      overall_total += reward
      if done:
        break
  average_reward = overall_total / number_of_games
  print("Average reward is {0}".format(average_reward))

play(20)

# output: predict next action
# input: observation, 
# accumulator with score

# train on pairs nearest start the most to least one the ones nearest the end
# train on pairs in longer sequences more 
# for the last pair train on the opposite action
# minimize a large score

