import tensorflow as tf
import numpy as np
import pdb

labels = [ [1,0,0] ]
logits = [ [1,-1,-1] ]
model_labels = tf.placeholder(tf.float32, shape=(1, 3))
model_logits = tf.placeholder(tf.float32, shape=(1, 3))
sess = tf.Session()
placeholders = { model_labels: labels, model_logits: logits }
sm = sess.run(tf.nn.softmax(model_logits), placeholders)
print("sm={0}".format(sm))
smce = sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=model_labels, logits=model_logits), placeholders)
print("smce={0}".format(smce))

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

print(softmax([1,0,0]))


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
    return ce

predictions = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.96]])
targets = np.array([[0,0,0,1],
                   [0,0,0,1]])
pdb.set_trace()
ans = 0.71355817782  #Correct answer
x = cross_entropy(predictions, targets)
print(np.isclose(x,ans))
