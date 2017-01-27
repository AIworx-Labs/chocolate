
import unittest
from unittest.mock import MagicMock

from chocolate.space import *
from chocolate.base import SearchAlgorithm

class TestAlgorithmMixin(unittest.TestCase):
    def setUp(self):
        self.mock_conn = MagicMock(name="connection")
        self.mock_space = MagicMock(name="space")

    def test_space_none_none(self):
        self.mock_conn.get_space.return_value = None
        self.assertRaises(RuntimeError, SearchAlgorithm, self.mock_conn, None)

    def test_space_not_equal_nowrite(self):
        s1 = Space({"a" : uniform(1, 2)})
        s2 = Space({"a" : uniform(1, 3)})
        
        self.mock_conn.get_space.return_value = s1
        self.assertRaises(RuntimeError, SearchAlgorithm, self.mock_conn, s2)

    def test_space_not_equal_write(self):
        s1 = Space({"a" : uniform(1, 2)})
        s2 = Space({"a" : uniform(1, 3)})
        
        self.mock_conn.get_space.return_value = s1
        algo = SearchAlgorithm(self.mock_conn, s2, clear_db=True)

        self.mock_conn.clear.assert_called_with()
        self.mock_conn.insert_space.assert_called_with(s2)
        self.assertEqual(algo.space, s2)

    def test_space_none_not_none(self):
        s1 = Space({"a" : uniform(1, 2)})
        
        self.mock_conn.get_space.return_value = None
        algo = SearchAlgorithm(self.mock_conn, s1)

        self.mock_conn.insert_space.assert_called_with(s1)
        self.assertEqual(algo.space, s1)

    def test_space_not_none_none(self):
        s1 = Space({"a" : uniform(1, 2)})
        
        self.mock_conn.get_space.return_value = s1
        algo = SearchAlgorithm(self.mock_conn, None)

        self.assertEqual(algo.space, s1)

    def test_update_value(self):
        token = {"a" : 0}
        algo = SearchAlgorithm(self.mock_conn, None)

        algo.update(token, 9.0)

        expected = {"_loss" : 9.0}
        self.mock_conn.update_result.assert_called_with(token, expected)

    def test_update_mapping(self):
        token = {"a" : 0}
        algo = SearchAlgorithm(self.mock_conn, None)

        algo.update(token, {"_loss" : 9.0})

        expected = {"_loss" : 9.0}
        self.mock_conn.update_result.assert_called_with(token, expected)
