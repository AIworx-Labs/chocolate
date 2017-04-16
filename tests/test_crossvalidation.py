import unittest
from unittest.mock import MagicMock

from chocolate.conditional import ThompsonSampling
from chocolate.space import Space, uniform
from chocolate.sample import *
from chocolate.search import *
from chocolate.crossvalidation import Repeat

class TestSearchRepeat(unittest.TestCase):
    def test_wrap_connection(self):
        mock_conn = MagicMock(name="connection")

        cv = Repeat(3)
        cv.wrap_connection(mock_conn)

        self.assertEqual(cv.all_results, mock_conn.all_results)
        self.assertEqual(cv.count_results, mock_conn.count_results)

    def test_all_results(self):
        mock_conn = MagicMock(name="connection")
        space = Space({"a": uniform(1, 2)})
        repeat_col_name = "rep"

        data = [{"a": 0, "_chocolate_id": 0, repeat_col_name: 0, "_loss": 0.3},
                {"a": 0, "_chocolate_id": 0, repeat_col_name: 1, "_loss": 0.3},
                {"a": 0, "_chocolate_id": 0, repeat_col_name: 2, "_loss": 0.4},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 0, "_loss": 0.3},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 1, "_loss": 0.4},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 2, "_loss": 0.5}]

        mock_conn.all_results.return_value = data
        mock_conn.count_results = len(data)
        mock_conn.pop_id = lambda x: x

        cv = Repeat(3, rep_col=repeat_col_name)
        cv.wrap_connection(mock_conn)
        cv.space = space

        self.assertEqual(cv.count_results(), 2)

        results = cv.all_results()
        for entry in results:
            if entry["_chocolate_id"] == 0:
                self.assertAlmostEqual(entry["_loss"], 1/3)
            elif entry["_chocolate_id"] == 1:
                self.assertAlmostEqual(entry["_loss"], 0.4)
    
    def test_next_new(self):
        mock_conn = MagicMock(name="connection")
        space = Space({"a": uniform(1, 2)})
        repeat_col_name = "rep"

        data = [{"a": 0, "_chocolate_id": 0, repeat_col_name: 0, "_loss": 0.3},
                {"a": 0, "_chocolate_id": 0, repeat_col_name: 1, "_loss": 0.3},
                {"a": 0, "_chocolate_id": 0, repeat_col_name: 2, "_loss": 0.4},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 0, "_loss": 0.3},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 1, "_loss": 0.4},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 2, "_loss": 0.5}]

        mock_conn.all_results.return_value = data
        mock_conn.count_results = len(data)
        mock_conn.pop_id = lambda x: x
        mock_conn.get_space.return_value = space

        cv = Repeat(3, rep_col=repeat_col_name)
        s = Random(mock_conn, space, cv)

        token, params = s.next()
        self.assertEqual(token["_chocolate_id"], 2)
    
    def test_next_repeat(self):
        mock_conn = MagicMock(name="connection")
        space = Space({"a": uniform(1, 2)})
        repeat_col_name = "rep"

        data = [{"a": 0, "_chocolate_id": 0, repeat_col_name: 0, "_loss": 0.3},
                {"a": 0, "_chocolate_id": 0, repeat_col_name: 1, "_loss": 0.3},
                {"a": 0, "_chocolate_id": 0, repeat_col_name: 2, "_loss": 0.4},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 0, "_loss": 0.3},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 1, "_loss": 0.4}]

        mock_conn.all_results.return_value = data
        mock_conn.count_results = len(data)
        mock_conn.pop_id = lambda x: x
        mock_conn.get_space.return_value = space

        cv = Repeat(3, rep_col=repeat_col_name)
        s = Random(mock_conn, space, cv)

        token, params = s.next()
        self.assertEqual(token["_chocolate_id"], 1)
    
    def test_next_repeat_with_nones(self):
        mock_conn = MagicMock(name="connection")
        space = Space({"a": uniform(1, 2)})
        repeat_col_name = "rep"

        data = [{"a": 0, "_chocolate_id": 0, repeat_col_name: 0, "_loss": 0.3},
                {"a": 0, "_chocolate_id": 0, repeat_col_name: 1, "_loss": 0.3},
                {"a": 0, "_chocolate_id": 0, repeat_col_name: 2, "_loss": 0.4},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 0, "_loss": None},
                {"a": 0.6, "_chocolate_id": 1, repeat_col_name: 1, "_loss": None}]

        mock_conn.all_results.return_value = data
        mock_conn.count_results = len(data)
        mock_conn.pop_id = lambda x: x
        mock_conn.get_space.return_value = space

        cv = Repeat(3, rep_col=repeat_col_name)
        s = Random(mock_conn, space, cv)

        token, params = s.next()
        self.assertEqual(token["_chocolate_id"], 1)
        self.assertEqual(token[repeat_col_name], 2)
        self.assertEqual(params["a"], 0.6 + (2 - 1))

# TODO: This should be a base SearchAlgorithm test
class TestRepeatThompson(unittest.TestCase):
    def setUp(self):
        self.repeat_col_name = "rep"
        self.mock_conn = MagicMock(name="connection")
        self.space = Space({"a": uniform(1, 2)})
        self.data = [{"a": 0, "_chocolate_id": 0, "_arm_id": 0, self.repeat_col_name: 0, "_loss": 0.3},
                     {"a": 0, "_chocolate_id": 0, "_arm_id": 0, self.repeat_col_name: 1, "_loss": 0.3},
                     {"a": 0, "_chocolate_id": 0, "_arm_id": 0, self.repeat_col_name: 2, "_loss": 0.4},
                     {"a": 0.6, "_chocolate_id": 1, "_arm_id": 0, self.repeat_col_name: 0, "_loss": 0.4},
                     {"a": 0.6, "_chocolate_id": 1, "_arm_id": 0, self.repeat_col_name: 1, "_loss": 0.5}]
        
        self.mock_conn.all_results.return_value = self.data
        self.mock_conn.count_results = len(self.data)
        self.mock_conn.pop_id = lambda x: x
        self.mock_conn.get_space.return_value = self.space

    def test_next(self):
        cv = Repeat(3, rep_col=self.repeat_col_name)
        s = ThompsonSampling(CMAES, self.mock_conn, self.space, cv)

        token, params = s.next()

        self.assertEqual(token["_chocolate_id"], 1)
        self.assertEqual(token[self.repeat_col_name], 2)
        self.assertEqual(params["a"], 0.6 + (2 - 1))