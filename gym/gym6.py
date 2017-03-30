# This POS code still does not work. It is way worse than random

import gym
import pdb
import tensorflow as tf
import numpy as np
import random

#env = gym.make('Breakout-v0')
env = gym.make('CartPole-v0')

number_of_actions = env.action_space.n
number_of_observations = env.observation_space.shape[0]
number_of_inputs = number_of_observations

q_learning_rate = 0.01
optimizer_learning_rate = 0.05

model_inputs = tf.placeholder(tf.float32, shape=(None, number_of_inputs), name="Inputs")
model_y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")

def model1():
  model_w = tf.get_variable("w", shape=(number_of_inputs, number_of_actions), dtype=tf.float32)
  model_b = tf.get_variable("b", shape=(number_of_actions), dtype=tf.float32)
  model_logits = tf.matmul(model_inputs, model_w) + model_b
  return model_logits

def model2():
  num_hidden1 = 10

  model_w1 = tf.get_variable("w1", shape=(number_of_inputs, num_hidden1), dtype=tf.float32)
  model_b1 = tf.get_variable("b1", shape=(num_hidden1), dtype=tf.float32)

  model_w2 = tf.get_variable("w2", shape=(num_hidden1, number_of_actions), dtype=tf.float32)
  model_b2 = tf.get_variable("b2", shape=(number_of_actions), dtype=tf.float32)

  hidden1 = tf.nn.relu(tf.matmul(model_inputs, model_w1) + model_b1)
  hidden2 = tf.matmul(hidden1, model_w2) + model_b2

  model_logits = hidden2
  return model_logits

model_logits = model2()

model_loss = tf.reduce_mean(tf.squared_difference(model_y, tf.reduce_max(model_logits, axis=1, keep_dims=True)))
#model_optimizer = tf.train.AdamOptimizer(optimizer_learning_rate)
model_optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
model_train_op = model_optimizer.minimize(model_loss)

class TrainingData:
  def __init__(self):
    self.training_data = []
    self.rewards = {}

  def size(self):
    return len(self.training_data)

  def add(self, training_data, reward):
    if reward in self.rewards:
      self.rewards[reward] += 1
    else:
      self.rewards[reward] = 1

    for tuple in training_data:
      self.training_data.append( (tuple[0], tuple[1], tuple[2], tuple[3], reward) )

    self.remove_lowest()

  def print_histogram(self):
    for reward in self.rewards:
      print("Reward {0} - Count: {1}".format(reward, self.rewards[reward]))

  def remove_lowest(self):
    if len(self.rewards) < 10:
      return

    lowest = np.amin(list(self.rewards.keys()))
    del(self.rewards[lowest])
    self.training_data = [ td for td in self.training_data if td[4] > lowest ]
    
  def start(self, batch_size):
    self.batch_size = batch_size
    self.next_i = 0

  def done(self):
    return self.next_i + self.batch_size > len(self.training_data)

  def next_batch(self, inputs, labels):
    sample_data = self.training_data.copy()
    random.shuffle(sample_data)
    for i in range(self.batch_size):
      td = sample_data[i+self.next_i]
      inputs[i] = td[0]
      if td[3] is None:
        y = td[2]
      else:
        y = td[2] + q_learning_rate * np.amax(session.run(model_logits, feed_dict = { model_inputs: [td[3]] }))
      labels[i] = y
        
    self.next_i += self.batch_size
  
def train():
  training_data = TrainingData()
  number_of_episodes = 250
  number_of_steps = 100
  minibatch_size = 32
  number_of_frames = number_of_episodes * number_of_steps
  decrement = (0.9 / number_of_frames) * 5
  epsilon = 1.0
  for i_episode in range(number_of_episodes):
    observation = env.reset()
    episode_data = []
    total_reward = 0
    for t in range(number_of_steps):
      epsilon -= decrement
      if epsilon < 0.1:
        epsilon = 0.1

      env.render()
      if random.random() < epsilon:
        action = env.action_space.sample()
      else:
        feed_dict = { model_inputs: [observation] }
        logits = session.run(model_logits, feed_dict)
        action = np.argmax(logits)
        
      next_observation, reward, done, info = env.step(action)
      total_reward += reward
      if done:
        episode_data.append((observation, action, reward, None))
        break

      episode_data.append((observation, action, reward, next_observation))
      observation = next_observation
     
    # train

    if training_data.size() > minibatch_size*2:
      training_data.start(minibatch_size)
      inputs = np.zeros((minibatch_size, number_of_inputs))
      y = np.zeros((minibatch_size, 1))
      training_data.next_batch(inputs, y)

      feed_dict = { model_inputs: inputs, model_y: y }
      _, loss = session.run([model_train_op, model_loss], feed_dict = feed_dict )
      print("Episode: {0}, Loss: {1}, Epsilon: {2} Reward: {3}".format(i_episode, loss, epsilon, total_reward))

    training_data.add(episode_data, total_reward)

  training_data.print_histogram()

session = tf.Session()
session.run(tf.global_variables_initializer())

train()

number_of_plays = 20

def play(number_of_episodes):
  env.reset()
  overall_total = 0
  for i_episode in range(number_of_episodes):
    observation = env.reset()
    for t in range(100):
      env.render()
      feed_dict = {model_inputs: [observation]}
      logits = session.run([model_logits], feed_dict = feed_dict)
      action = np.argmax(logits)

      observation, reward, done, info = env.step(action)
      overall_total += reward
      if done:
        break
  average_reward = overall_total / number_of_episodes
  print("Average reward is {0}".format(average_reward))

play(20)

