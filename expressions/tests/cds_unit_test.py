import unittest
import pdb
import tensorflow as tf
from cds1 import CDS

class TddTestCDS(unittest.TestCase):

  def setUp(self):
    self.session = tf.Session()

  def test_create(self):
    assert( CDS(self.session, 1,1) )

