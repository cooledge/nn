import unittest
import pdb
import tensorflow as tf
from cds1 import CDS

class TddTestCDS(unittest.TestCase):

  def setUp(self):
    self.session = tf.Session()

  def test_create(self):
    assert( CDS(self.session, 1,1) )

  def test_session_is_set(self):
    self.assertEqual(self.session, CDS(self.session, 1, 1).session)

  def test_length_is_set(self):
    self.assertEqual(23, CDS(self.session, 23, 1).length)

  def test_depth_is_set(self):
    self.assertEqual(42, CDS(self.session, 23, 42).depth)

  def test_current_empty(self):
    cds = CDS(self.session, 2, 4)
    self.session.run(tf.global_variables_initializer())
    self.assertEqual([], cds.current())
