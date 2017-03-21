from itertools import product
import unittest
from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import floats

from chocolate import QuasiRandom
from chocolate.space import Space, uniform


class TestQuasiRandom(unittest.TestCase):
    def setUp(self):
        self.space = {"a": uniform(1, 10),
                      "b": uniform(5, 15)}

        self.mock_conn = MagicMock(name="connection")
        self.mock_conn.get_space.return_value = Space(self.space)

        self.search = QuasiRandom(self.mock_conn, self.space)

    def test_cold_start(self):
        self.mock_conn.all_results.return_value = []
        self.mock_conn.count_results.return_value = 0

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 0)

        self.assertIn("a", p)

    def test_bootstrap_low(self):
        db = [{"_chocolate_id": 0, "a": 0, "b": 0, "_loss": 0.1},
              {"_chocolate_id": 1, "a": 0.5, "b": 0.5, "_loss": 0.1}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 2)

        self.assertIn("a", p)

    def test_bootstrap_high(self):
        db = [{"_chocolate_id": i, "a": i * (1.0 / 30), "b": i * (1.0 / 30), "_loss": i * (1.0 / 30)} for i in
              range(30)]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], len(db))

        self.assertIn("a", p)

    def test_bootstrap_all_none(self):
        db = [{"_chocolate_id": 0, "a": 0, "b": 0, "_loss": None},
              {"_chocolate_id": 1, "a": 0.5, "b": 0.5, "_loss": None},
              {"_chocolate_id": 2, "a": 0.0, "b": 0.5, "_loss": None},
              {"_chocolate_id": 3, "a": 0.5, "b": 0.0, "_loss": None},
              {"_chocolate_id": 4, "a": 0.9, "b": 0.0, "_loss": None}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 5)

        self.assertIn("a", p)
        self.assertGreaterEqual(p["a"], 1)
        self.assertLess(p["a"], 10)

        res = self.mock_conn.insert_result.call_args
        self.assertIn("_chocolate_id", res[0][0])
        self.assertEqual(res[0][0]["_chocolate_id"], 5)
        self.assertGreaterEqual(res[0][0]["a"], 0)
        self.assertLess(res[0][0]["a"], 1)

    @given(floats(allow_nan=True, allow_infinity=True))
    def test_losses(self, f):
        db = [{"_chocolate_id": 0, "a": 0.9, "b": 0.0, "_loss": 6},
              {"_chocolate_id": 1, "a": 0.9, "b": 0.0, "_loss": 7},
              {"_chocolate_id": 2, "a": 0.9, "b": 0.0, "_loss": f}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)

        # Hypothesis does not call setUp/tearDown
        self.search.rndrawn = 0

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 3)
        self.assertIn("a", p)
        self.assertGreaterEqual(p["a"], 1)
        self.assertLess(p["a"], 10)

        res = self.mock_conn.insert_result.call_args
        self.assertIn("_chocolate_id", res[0][0])
        self.assertEqual(res[0][0]["_chocolate_id"], 3)
        self.assertGreaterEqual(res[0][0]["a"], 0)
        self.assertLess(res[0][0]["a"], 1)

    def test_permutation_ea(self):
        self.mock_conn.all_results.return_value = []
        self.mock_conn.count_results.return_value = 0

        search = QuasiRandom(self.mock_conn, self.space, permutations="ea")
        token, p = search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 0)

        self.assertIn("a", p)

    def test_permutations(self):
        self.mock_conn.all_results.return_value = []
        self.mock_conn.count_results.return_value = 0

        search = QuasiRandom(self.mock_conn, self.space, permutations=((0, 1), (0, 2, 1)))
        token, p = search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 0)

        self.assertIn("a", p)

    def test_seed(self):
        self.mock_conn.all_results.return_value = []
        self.mock_conn.count_results.return_value = 0

        search = QuasiRandom(self.mock_conn, self.space, seed=42)
        token, p = search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 0)

        self.assertIn("a", p)
