# input is observation and previous action
import gym
import pdb
import tensorflow as tf
import numpy as np
import random

env = gym.make('CartPole-v0')

random_action_probability = 0.5
number_of_actions = env.action_space.n
number_of_observations = env.observation_space.shape[0]
number_of_inputs = number_of_observations

model_inputs = tf.placeholder(tf.float32, shape=(None, number_of_inputs), name="Inputs")
model_y = tf.placeholder(tf.float32, shape=(None, number_of_actions), name="Y")
model_w = tf.get_variable("w", shape=(number_of_inputs, number_of_actions), dtype=tf.float32)
model_b = tf.get_variable("b", shape=(number_of_actions), dtype=tf.float32)
model_logits = tf.matmul(model_inputs, model_w) + model_b
# (none, number_of_actions)
model_predict = tf.nn.softmax(model_logits)
'''
model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_actions, logits=model_logits))
'''

model_optimizer = tf.train.AdamOptimizer(0.01)
model_train_op = model_optimizer.minimize(model_loss)

class TrainingData:
  def __init__(self):
    self.training_data = []

  def add(self, reward, observations, actions):
    for i in range(len(actions)): 
      if i == 0:
        next
      observation = observations[i]
      action = actions[i]
      last_action = actions[i-1]

      input = np.zeros((number_of_inputs))
      for j in range(len(observation)):
        input[j] = observation[j]
      input[j+1+action] = 1
      self.training_data += [(input, action)]

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
   
def train():
  training_data = TrainingData()
  number_of_episodes = 20
  number_of_steps = 100
  for i_episode in range(number_of_episodes):
    for t in range(number_of_steps):
      env.render()
      if random.random() < random_action_probability:
        action = env.action_space.sample()
      else:
        
      observation, reward, done, info = env.step(action)
      actions += [action]
      total_reward += reward
      if done:
        training_data.add(total_reward, observations, actions)
        break
      observations += [observation]
    overall_total += total_reward

session = tf.Session()
session.run(tf.global_variables_initializer())

epochs = 20
for epoch in range(epochs):
  training_data.shuffle()
  batch_size = 20
  training_data.start(batch_size)
  while not training_data.done():
    inputs = np.zeros((batch_size, number_of_inputs))
    labels = np.zeros((batch_size, number_of_actions))
    training_data.done()
    training_data.next_batch(inputs, labels)
    feed_dict = {model_inputs: inputs, model_actions: labels}
    loss, _ = session.run([model_loss, model_train_op], feed_dict = feed_dict)
  print("Epoch {0} Loss is {1}".format(epoch, loss))

number_of_plays = 20

def play(number_of_games):
  env.reset()
  overall_total = 0
  for i_episode in range(number_of_games):
    observation = env.reset()
    last_action = 0
    for t in range(100):
      env.render()
      #action = env.action_space.sample()

      inputs = np.zeros((number_of_inputs))
      for j in range(len(observation)):
        inputs[j] = observation[j]
      if t == 0:
        inputs[j+1+last_action] = 0.5
        inputs[j+2+last_action] = 0.5
      else:
        inputs[j+1+last_action] = 1
      
      feed_dict = {model_inputs: [inputs]}
      predict = session.run([model_predict], feed_dict = feed_dict)
      action = np.argmax(predict)
      last_action = action

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

