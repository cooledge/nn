import tensorflow as tf
import numpy as np
import pdb

tf.enable_eager_execution()

sample = tf.constant([ [1] ])

def identity(board):
  return tf.constant([1,2])

def works(sample):
  print(tf.map_fn(identity, sample))
 
works(sample) 
pdb.set_trace()
'''
OUTPUTS: tf.Tensor([[1 2]], shape=(1, 2), dtype=int32)
'''

def not_working(sample):
  print(tf.map_fn(identity, sample, dtype=tf.int32))

not_working(sample)

''' OUTPUTS: 
Traceback (most recent call last):
  File "/home/dev/tf_1.19/tf_1.19/lib/python3.5/site-packages/tensorflow/python/util/nest.py", line 179, in assert_same_structure
      _pywrap_tensorflow.AssertSameStructure(nest1, nest2, check_types)
  ValueError: The two structures don't have the same nested structure.

  First structure: type=tuple str=(tf.int32,)

  Second structure: type=EagerTensor str=tf.Tensor([1 2], shape=(2,), dtype=int32)

  More specifically: Substructure "type=tuple str=(tf.int32,)" is a sequence, while substructure "type=EagerTensor str=tf.Tensor([1 2], shape=(2,), dtype=int32)" is not

  During handling of the above exception, another exception occurred:

  Traceback (most recent call last):
    File "e.py", line 22, in <module>
      not_working(sample)
    File "e.py", line 20, in not_working
      print(tf.map_fn(identity, sample, dtype=(tf.int32,)))
    File "/home/dev/tf_1.19/tf_1.19/lib/python3.5/site-packages/tensorflow/python/ops/functional_ops.py", line 497, in map_fn
      maximum_iterations=n)
    File "/home/dev/tf_1.19/tf_1.19/lib/python3.5/site-packages/tensorflow/python/ops/control_flow_ops.py", line 3532, in while_loop
      loop_vars = body(*loop_vars)
    File "/home/dev/tf_1.19/tf_1.19/lib/python3.5/site-packages/tensorflow/python/ops/control_flow_ops.py", line 3525, in <lambda>
      body = lambda i, lv: (i + 1, orig_body(*lv))
    File "/home/dev/tf_1.19/tf_1.19/lib/python3.5/site-packages/tensorflow/python/ops/functional_ops.py", line 487, in compute
      nest.assert_same_structure(dtype or elems, packed_fn_values)
    File "/home/dev/tf_1.19/tf_1.19/lib/python3.5/site-packages/tensorflow/python/util/nest.py", line 186, in assert_same_structure
      % (str(e), str1, str2))
  ValueError: The two structures don't have the same nested structure.

  First structure: type=tuple str=(tf.int32,)

  Second structure: type=EagerTensor str=tf.Tensor([1 2], shape=(2,), dtype=int32)

  More specifically: Substructure "type=tuple str=(tf.int32,)" is a sequence, while substructure "type=EagerTensor str=tf.Tensor([1 2], shape=(2,), dtype=int32)" is not
  Entire first structure:
  (.,)
  Entire second structure:
  .
'''
