import unittest
from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import floats

from chocolate import ThompsonSampling
from chocolate.space import Space, uniform, quantized_uniform


class TestThompsonSampling(unittest.TestCase):
    def setUp(self):
        self.space = [{"cond": 1, "a": uniform(1, 10), "b": uniform(5, 15)},
                      {"cond": 2, "x": quantized_uniform(1, 10, 1), "y": quantized_uniform(5, 15, 1)}]

        self.mock_conn = MagicMock(name="connection")
        self.mock_conn.get_space.return_value = Space(self.space)

        self.mock_algo = MagicMock(name="algorithm")

        mock_algo_inst = MagicMock(name="algo_instance")
        mock_algo_inst.next.return_value = {"_chocolate_id": 77}, {'cond_1_b': 8, 'cond_1_a': 5, '_subspace': 0.0}

        self.mock_algo.return_value = mock_algo_inst

    def test_init(self):
        ThompsonSampling(self.mock_algo, self.mock_conn, self.space)
        self.assertEqual(self.mock_algo.call_count, 2)

    @given(floats(allow_nan=True, allow_infinity=True))
    def test_next(self, f):
        db = [{"_chocolate_id": 21, "_arm_id": 1, "_subspace": 0.5, "cond_1_b": None, "cond_1_a": None,
               "cond_2_x": 0.333, "cond_2_y": 0.2, "_loss": f},
              {"_chocolate_id": 0, "_arm_id": 0, "_subspace": 0.5, "cond_1_b": 0.32, "cond_1_a": 0.29,
               "cond_2_x": None, "cond_2_y": None, "_loss": f}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)

        search = ThompsonSampling(self.mock_algo, self.mock_conn, self.space)
        token, p = search.next()

        self.assertIn("_arm_id", token)
        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 77)
        self.assertIn("a", p)
        self.assertIn("b", p)
        self.assertEqual(p["a"], 5)
        self.assertEqual(p["b"], 8)
