import unittest
import pdb
import tensorflow as tf
from cds import CDS
from expressions import Evaluator

class TddTestExpressions(unittest.TestCase):

  def setUp(self):
    self.session = tf.Session()
    self.evaluator = Evaluator()

  def test_expressions1():
    expression1 = self.evaluator.evaluate("move the boat to the island")
    expected1 = [{'action': 'move_chess_piece', 'piece': {'thing': 'boat', 'determiner': 'the'}, 'square': {'thing': 'island', 'determiner': 'the'}}]
    assert expression1 == expected1

  def test_expressions2():
    expression2 = self.evaluator.evaluate("move t1 to t2")
    expected2 = [{'action': 'move_chess_piece', 'piece': 't1', 'square': 't2'}]
    assert expression2 == expected2

