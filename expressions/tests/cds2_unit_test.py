import unittest
import pdb
import tensorflow as tf
from cds2 import CDS

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
    cds.initialize(["l0_c0"])
    self.assertEqual([((0, 0), "l0_c0")], cds.current())

  def test_current_joins_initialize_two(self):
    cds = CDS(self.session, 2, 1)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1"])
    self.assertEqual([((0, 0), "l0_c0"), ((0, 1), "l0_c1")], cds.current())

  def test_current_joins_initialize_three(self):
    cds = CDS(self.session, 3, 1)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    self.assertEqual([((0, 0), "l0_c0"), ((0, 1), "l0_c1"), ((0, 2), "l0_c2")], cds.current())

  def check_current_joins_three_one_edge_case1(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0)], (next_row, 0), "next")
    self.assertEqual([((next_row, 0), "next"), ((0, 1), "l0_c1"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_one_edge_case1_next_1(self):
    self.check_current_joins_three_one_edge_case1(1)

  def test_current_joins_three_one_edge_case1_next_2(self):
    self.check_current_joins_three_one_edge_case1(2)

  def check_current_joins_three_one_edge_case2(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 1)], (next_row, 0), "next")
    self.assertEqual([((0, 0), "l0_c0"), ((next_row, 0), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_one_edge_case2_next_1(self):
    self.check_current_joins_three_one_edge_case2(1)

  def test_current_joins_three_one_edge_case2_next_2(self):
    self.check_current_joins_three_one_edge_case2(2)

  def check_current_joins_three_one_edge_case3(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 2)], (next_row, 0), "next")
    self.assertEqual([((0, 0), "l0_c0"), ((next_row, 0), "next"), ((0, 1), "l0_c1")], cds.current())

  def test_current_joins_three_one_edge_case3_next_1(self):
    self.check_current_joins_three_one_edge_case3(1)

  def test_current_joins_three_one_edge_case3_next_1(self):
    self.check_current_joins_three_one_edge_case3(2)

  def check_current_joins_three_one_edge_case4(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0)], (next_row, 1), "next")
    self.assertEqual([((0, 1), "l0_c1"), ((next_row, 1), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_one_edge_case4(self):
    self.check_current_joins_three_one_edge_case4(1)

  def test_current_joins_three_one_edge_case4(self):
    self.check_current_joins_three_one_edge_case4(2)

  def check_current_joins_three_one_edge_case5(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 1)], (next_row, 1), "next")
    self.assertEqual([((0, 0), "l0_c0"), ((next_row, 1), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_one_edge_case5_next_1(self):
    self.check_current_joins_three_one_edge_case5(1)

  def test_current_joins_three_one_edge_case5_next_2(self):
    self.check_current_joins_three_one_edge_case5(2)

  def check_current_joins_three_one_edge_case6(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 2)], (next_row, 1), "next")
    self.assertEqual([((0, 0), "l0_c0"), ((0, 1), "l0_c1"), ((next_row, 1), "next")], cds.current())

  def test_current_joins_three_one_edge_case6_next_1(self):
    self.check_current_joins_three_one_edge_case6(1)

  def test_current_joins_three_one_edge_case6_next_2(self):
    self.check_current_joins_three_one_edge_case6(2)

  def check_current_joins_three_one_edge_case7(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0)], (next_row, 2), "next")
    self.assertEqual([((0, 1), "l0_c1"), ((0, 2), "l0_c2"), ((next_row, 2), "next")], cds.current())

  def test_current_joins_three_one_edge_case7_next_1(self):
    self.check_current_joins_three_one_edge_case7(1)

  def test_current_joins_three_one_edge_case7_next_2(self):
    self.check_current_joins_three_one_edge_case7(2)

  def check_current_joins_three_one_edge_case8(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 1)], (next_row, 2), "next")
    self.assertEqual([((0, 0), "l0_c0"), ((0, 2), "l0_c2"), ((next_row, 2), "next")], cds.current())

  def test_current_joins_three_one_edge_case8_next_1(self):
    self.check_current_joins_three_one_edge_case8(1)

  def test_current_joins_three_one_edge_case8_next_2(self):
    self.check_current_joins_three_one_edge_case8(2)

  def check_current_joins_three_one_edge_case9(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 2)], (next_row, 2), "next")
    self.assertEqual([((0, 0), "l0_c0"), ((0, 1), "l0_c1"), ((next_row, 2), "next")], cds.current())

  def test_current_joins_three_one_edge_case9_next_1(self):
    self.check_current_joins_three_one_edge_case9(1)

  def test_current_joins_three_one_edge_case9_next_2(self):
    self.check_current_joins_three_one_edge_case9(2)

  def check_current_joins_three_two_edges_case1(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0), (0, 1)], (next_row, 0), "next")
    self.assertEqual([((next_row, 0), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_two_edges_case1_next_1(self):
    self.check_current_joins_three_two_edges_case1(1)

  def test_current_joins_three_two_edges_case1_next_2(self):
    self.check_current_joins_three_two_edges_case1(2)

  def p2s(tuple):
    return "l{0}_c{1}".format(tuple[0], tuple[1]);

  def check_current_joins_three_two_edges_case1_next_2_l1_cN(self, l0_col_second, l0_col, l1_col):
    next_row = 2

    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, l0_col)], (1, l1_col), "l1_c0")
    cds.joins([(0, l0_col_second), (1, l1_col)], (next_row, 0), "next")
    return cds

  def test_current_joins_three_two_edges_case1_next_2_l0_c1_to_l1_c0(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(0, 1, 0)
    self.assertEqual([((2, 0), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_l0_c1_to_l1_c1(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(0, 1, 1)
    self.assertEqual([((2, 0), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_l0_c1_to_l1_c2(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(0, 1, 2)
    self.assertEqual([((2, 0), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_l0_c2_to_l1_c0(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(0, 2, 0)
    self.assertEqual([((2, 0), "next"), ((0, 1), "l0_c1")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_l0_c2_to_l1_c1(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(0, 2, 1)
    self.assertEqual([((2, 0), "next"), ((0, 1), "l0_c1")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_l0_c2_to_l1_c2(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(0, 2, 2)
    self.assertEqual([((2, 0), "next"), ((0, 1), "l0_c1")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l1_l0_c0_to_l1_c0(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(1, 0, 0)
    self.assertEqual([((2, 0), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l1_l0_c0_to_l1_c1(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(1, 0, 1)
    self.assertEqual([((2, 0), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l1_l0_c0_to_l1_c2(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(1, 0, 2)
    self.assertEqual([((2, 0), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l1_l0_c2_to_l1_c0(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(1, 2, 0)
    self.assertEqual([((0, 0), "l0_c0"), ((2, 0), "next")], cds.current())
  
  def test_current_joins_three_two_edges_case1_next_2_second_l1_l0_c2_to_l1_c1(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(1, 2, 1)
    self.assertEqual([((0, 0), "l0_c0"), ((2, 0), "next")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l1_l0_c2_to_l1_c2(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(1, 2, 2)
    self.assertEqual([((0, 0), "l0_c0"), ((2, 0), "next")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l2_l0_c0_to_l1_c0(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(2, 0, 0)
    self.assertEqual([((2, 0), "next"), ((0, 1), "l0_c1")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l2_l0_c0_to_l1_c1(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(2, 0, 1)
    self.assertEqual([((2, 0), "next"), ((0, 1), "l0_c1")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l2_l0_c0_to_l1_c2(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(2, 0, 2)
    self.assertEqual([((2, 0), "next"), ((0, 1), "l0_c1")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l2_l0_c2_to_l1_c0(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(2, 1, 0)
    self.assertEqual([((0, 0), "l0_c0"), ((2, 0), "next")], cds.current())
  
  def test_current_joins_three_two_edges_case1_next_2_second_l2_l0_c2_to_l1_c1(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(2, 1, 1)
    self.assertEqual([((0, 0), "l0_c0"), ((2, 0), "next")], cds.current())

  def test_current_joins_three_two_edges_case1_next_2_second_l2_l0_c2_to_l1_c2(self):
    cds = self.check_current_joins_three_two_edges_case1_next_2_l1_cN(2, 1, 2)
    self.assertEqual([((0, 0), "l0_c0"), ((2, 0), "next")], cds.current())

  def check_current_joins_three_two_edges_case2(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0), (0, 1)], (next_row, 1), "next")
    self.assertEqual([((next_row, 1), "next"), ((0, 2), "l0_c2")], cds.current())

  def test_current_joins_three_two_edges_case2_next_1(self):
    self.check_current_joins_three_two_edges_case2(1)

  def test_current_joins_three_two_edges_case2_next_2(self):
    self.check_current_joins_three_two_edges_case2(2)

  def check_current_joins_three_two_edges_case3(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0), (0, 1)], (next_row, 2), "next")
    self.assertEqual([((0, 2), "l0_c2"), ((next_row, 2), "next")], cds.current())

  def test_current_joins_three_two_edges_case3_next_1(self):
    self.check_current_joins_three_two_edges_case3(1)

  def test_current_joins_three_two_edges_case3_next_2(self):
    self.check_current_joins_three_two_edges_case3(2)

  def check_current_joins_three_two_edges_case4(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0), (0, 2)], (next_row, 0), "next")
    self.assertEqual([((next_row, 0), "next"), ((0, 1), "l0_c1")], cds.current())

  def test_current_joins_three_two_edges_case4_next_1(self):
    self.check_current_joins_three_two_edges_case4(1)

  def test_current_joins_three_two_edges_case4_next_2(self):
    self.check_current_joins_three_two_edges_case4(2)

  def check_current_joins_three_two_edges_case5(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0), (0, 2)], (next_row, 1), "next")
    self.assertEqual([((0, 1), "l0_c1"), ((next_row, 1), "next")], cds.current())

  def test_current_joins_three_two_edges_case5_next_1(self):
    self.check_current_joins_three_two_edges_case5(1)

  def test_current_joins_three_two_edges_case5_next_2(self):
    self.check_current_joins_three_two_edges_case5(2)

  def check_current_joins_three_two_edges_case6(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0), (0, 2)], (next_row, 2), "next")
    self.assertEqual([((0, 1), "l0_c1"), ((next_row, 2), "next")], cds.current())

  def test_current_joins_three_two_edges_case6_next_1(self):
    self.check_current_joins_three_two_edges_case6(1)

  def test_current_joins_three_two_edges_case6_next_2(self):
    self.check_current_joins_three_two_edges_case6(2)

  def check_current_joins_three_two_edges_case7(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 1), (0, 2)], (next_row, 0), "next")
    self.assertEqual([((0, 0), "l0_c0"), ((next_row, 0), "next")], cds.current())

  def test_current_joins_three_two_edges_case7_next_1(self):
    self.check_current_joins_three_two_edges_case7(1)

  def test_current_joins_three_two_edges_case7_next_2(self):
    self.check_current_joins_three_two_edges_case7(2)

  def check_current_joins_three_two_edges_case8(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 1), (0, 2)], (next_row, 1), "next")
    self.assertEqual([((0, 0), "l0_c0"), ((next_row, 1), "next")], cds.current())

  def test_current_joins_three_two_edges_case8_next_1(self):
    self.check_current_joins_three_two_edges_case8(1)

  def test_current_joins_three_two_edges_case8_next_2(self):
    self.check_current_joins_three_two_edges_case8(2)

  def check_current_joins_three_two_edges_case9(self, next_row):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 1), (0, 2)], (next_row, 2), "next")
    self.assertEqual([((0, 0), "l0_c0"), ((next_row, 2), "next")], cds.current())

  def test_current_joins_three_two_edges_case9_next_1(self):
    self.check_current_joins_three_two_edges_case9(1)

  def test_current_joins_three_two_edges_case9_next_2(self):
    self.check_current_joins_three_two_edges_case9(2)

  def check_current_joins_three_three_edges_case1(self, to_column):
    cds = CDS(self.session, 3, 2)
    self.session.run(tf.global_variables_initializer())
    cds.initialize(["l0_c0", "l0_c1", "l0_c2"])
    cds.joins([(0, 0), (0, 1), (0, 2)], (1, to_column), "next")
    self.assertEqual([((1, to_column), "next")], cds.current())

  def test_current_joins_three_three_edges_case1(self):
    self.check_current_joins_three_three_edges_case1(0)

  def test_current_joins_three_three_edges_case2(self):
    self.check_current_joins_three_three_edges_case1(1)

  def test_current_joins_three_three_edges_case3(self):
    self.check_current_joins_three_three_edges_case1(2)

