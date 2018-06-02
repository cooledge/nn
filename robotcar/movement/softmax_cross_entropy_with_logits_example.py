import numpy as np
import pdb

logits = [
  [
    [ 2.2215798, -6.969508 ,  1.1228983,  1.1228696,  2.2216043],
    [ 2.2215798, -6.969508 ,  1.1228983,  1.1228696,  2.2216043],
    [ 2.2215798, -6.969508 ,  1.1228983,  1.1228696,  2.2216043],
    [ 2.2215798, -6.969508 ,  1.1228983,  1.1228696,  2.2216043],
    [ 2.2215798, -6.969508 ,  1.1228983,  1.1228696,  2.2216043],
    [ 2.2215798, -6.969508 ,  1.1228983,  1.1228696,  2.2216043],
    [ 2.2215798, -6.969508 ,  1.1228983,  1.1228696,  2.2216043],
    [ 2.2215798, -6.969508 ,  1.1228983,  1.1228696,  2.2216043]
  ]
]

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

softmax_logits = [softmax(l) for l in logits[0]]
pdb.set_trace()
y = [
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1],
      [0, 0, 0, 0, 1],
      [0, 0, 0, 0, 1]
    ]

def cross_entropy(predictions, targets, epsilon=1e-12):
  """
      Computes cross entropy between targets (encoded as one-hot vectors)
      and predictions. 
      Input: predictions (N, k) ndarray
             targets (N, k) ndarray        
      Returns: scalar
  """
  predictions = np.array(predictions)
  N = predictions.shape[0]
  ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
  return ce

ce = cross_entropy(softmax_logits, y)
print(ce)
# losses = [2.2670712]
