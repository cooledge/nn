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
    self.assertEqual(1, CDS(self.session, 1, 1).length)

  def test_depth_is_set(self):
    self.assertEqual(1, CDS(self.session, 1, 1).depth)

  def test_current_empty(self):
    cds = CDS(self.session, 1, 1)
    self.session.run(tf.global_variables_initializer())
    self.assertEqual([], cds.current())

  def test_current_joins_initialize_one(self):
    cds = CDS(self.session, 1, 1)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one"])
    self.assertEqual([((0, 0), "one")], cds.current())

  def test_current_joins_initialize_two(self):
    cds = CDS(self.session, 2, 1)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two"])
    self.assertEqual([((0, 0), "one"), ((0, 1), "two")], cds.current())

  def test_current_joins_initialize_three(self):
    cds = CDS(self.session, 3, 1)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    self.assertEqual([((0, 0), "one"), ((0, 1), "two"), ((0, 2), "three")], cds.current())

  def test_current_joins_three_one_edge_case1(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0)], (1, 0), "next")
    self.assertEqual([((1, 0), "next"), ((0, 1), "two"), ((0, 2), "three")], cds.current())

  def test_current_joins_three_one_edge_case2(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 1)], (1, 0), "next")
    self.assertEqual([((0, 0), "one"), ((1, 0), "next"), ((0, 2), "three")], cds.current())

  def test_current_joins_three_one_edge_case3(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 2)], (1, 0), "next")
    self.assertEqual([((0, 0), "one"), ((1, 0), "next"), ((0, 1), "two")], cds.current())

  def test_current_joins_three_one_edge_case4(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0)], (1, 1), "next")
    self.assertEqual([((0, 1), "two"), ((1, 1), "next"), ((0, 2), "three")], cds.current())

  def test_current_joins_three_one_edge_case5(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 1)], (1, 1), "next")
    self.assertEqual([((0, 0), "one"), ((1, 1), "next"), ((0, 2), "three")], cds.current())

  def test_current_joins_three_one_edge_case6(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 2)], (1, 1), "next")
    self.assertEqual([((0, 0), "one"), ((0, 1), "two"), ((1, 1), "next")], cds.current())

  def test_current_joins_three_one_edge_case7(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0)], (1, 2), "next")
    self.assertEqual([((0, 1), "two"), ((0, 2), "three"), ((1, 2), "next")], cds.current())

  def test_current_joins_three_one_edge_case8(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 1)], (1, 2), "next")
    self.assertEqual([((0, 0), "one"), ((0, 2), "three"), ((1, 2), "next")], cds.current())

  def test_current_joins_three_one_edge_case9(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 2)], (1, 2), "next")
    self.assertEqual([((0, 0), "one"), ((0, 1), "two"), ((1, 2), "next")], cds.current())

  def test_current_joins_three_two_edges_case1(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0), (0, 1)], (1, 0), "next")
    self.assertEqual([((1, 0), "next"), ((0, 2), "three")], cds.current())

  def test_current_joins_three_two_edges_case2(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0), (0, 1)], (1, 1), "next")
    self.assertEqual([((1, 1), "next"), ((0, 2), "three")], cds.current())

  def test_current_joins_three_two_edges_case3(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0), (0, 1)], (1, 2), "next")
    self.assertEqual([((0, 2), "three"), ((1, 2), "next")], cds.current())

  def test_current_joins_three_two_edges_case4(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0), (0, 2)], (1, 0), "next")
    self.assertEqual([((1, 0), "next"), ((0, 1), "two")], cds.current())

  def test_current_joins_three_two_edges_case5(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0), (0, 2)], (1, 1), "next")
    self.assertEqual([((0, 1), "two"), ((1, 1), "next")], cds.current())

  def test_current_joins_three_two_edges_case6(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0), (0, 2)], (1, 2), "next")
    self.assertEqual([((0, 1), "two"), ((1, 2), "next")], cds.current())

  def test_current_joins_three_two_edges_case7(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 1), (0, 2)], (1, 0), "next")
    self.assertEqual([((0, 0), "one"), ((1, 0), "next")], cds.current())

  def test_current_joins_three_two_edges_case8(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 1), (0, 2)], (1, 1), "next")
    self.assertEqual([((0, 0), "one"), ((1, 1), "next")], cds.current())

  def test_current_joins_three_two_edges_case9(self):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 1), (0, 2)], (1, 2), "next")
    self.assertEqual([((0, 0), "one"), ((1, 2), "next")], cds.current())

  def check_current_joins_three_three_edges_case1(self, to_column):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two", "three"])
    cds.joins([(0, 0), (0, 1), (0, 2)], (1, to_column), "next")
    self.assertEqual([((1, to_column), "next")], cds.current())

  def test_current_joins_three_three_edges_case1(self):
    self.check_current_joins_three_three_edges_case1(0)

  def test_current_joins_three_three_edges_case2(self):
    self.check_current_joins_three_three_edges_case1(1)

  def test_current_joins_three_three_edges_case3(self):
    self.check_current_joins_three_three_edges_case1(2)

  '''
  def test_current_joins_set_one_one(self):
    cds = CDS(self.session, 5, 5)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one"])
    cds.joins([(0, 0)], (1, 0), "one_one")
    self.assertEqual([((1, 0), "one_one")], cds.current())

  def test_current_joins_set_one_one_one(self):
    cds = CDS(self.session, 5, 5)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one"])
    cds.joins([(0, 0)], (1, 0), "one_one")
    cds.joins([(1, 0)], (2, 0), "one_one_one")
    self.assertEqual([((2, 0), "one_one_one")], cds.current())

  def test_current_joins_set_two_one(self):
    cds = CDS(self.session, 5, 5)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two"])
    cds.joins([(0, 0)], (1, 0), "one_one")
    pdb.set_trace()
    self.assertEqual([((1, 0), "one_one"), ((0, 1), "two")], cds.current())

  def test_current_joins_set_two_two(self):
    cds = CDS(self.session, 5, 5)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["one", "two"])
    cds.joins([(0, 1)], (1, 1), "two_one")
    self.assertEqual([((0, 0), "one"), ((1, 1), "two_one")], cds.current())
  '''

