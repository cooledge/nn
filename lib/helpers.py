import pdb
import random
import numpy as np

def select_by_indexes(values, indexes):
  return np.array([values[index] for index in indexes])

def split_by_percentage(values, percentages):
  split = []
  sizes = [int(percentage*len(values)/100) for percentage in percentages]
  start = 0
  for size in sizes:
    split.append(np.array(values[start:start+size]))
    start += size
  return split

def splits_by_percentages(valuess, percentages):
  indexes = [index for index in range(len(valuess[0]))]
  random.shuffle(indexes)
  splits = split_by_percentage(indexes, percentages)
  new_valuess = []
  for values in valuess:
    new_values = []
    for split in splits:
      new_values.append(select_by_indexes(values, split))
    new_valuess.append(new_values)
  return new_valuess;

if __name__ == '__main__':
  pdb.set_trace()
#assert split_by_percentage([1,2,3,4,5,6,7,8,9,0], [20, 20, 30, 30]) == [np.array([1, 2]), np.array([3, 4]), np.array([5, 6, 7]), np.array([8, 9, 0])]
#assert split_by_percentage([1,2,3,4,5,6,7,8,9,0], [10, 20, 30, 40]) == [[1], [2, 3], [4, 5, 6], [7, 8, 9, 0]]

#print(splits_by_percentages([[1,2,3,4,5,6,7,8,9,0]], [10, 10, 80]))
#print(splits_by_percentages([[1,2,3,4,5,6,7,8,9,0], [1,2,3,4,5,6,7,8,9,0]], [10, 10, 80]))
