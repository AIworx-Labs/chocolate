from itertools import product
import unittest
from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import floats

from chocolate import Random
from chocolate.space import Space, uniform, quantized_uniform


class TestRandom(unittest.TestCase):
    def setUp(self):
        s = {"a": uniform(1, 10),
             "b": uniform(5, 15)}

        self.mock_conn = MagicMock(name="connection")
        self.mock_conn.get_space.return_value = Space(s)

        self.search = Random(self.mock_conn, s)

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


class TestDiscreteRandom(unittest.TestCase):
    def setUp(self):
        self.qa = quantized_uniform(0, 3, 1)
        self.qb = quantized_uniform(1, 4, 1)
        s = {"a": self.qa,
             "b": self.qb}

        self.mock_conn = MagicMock(name="connection")
        self.mock_conn.get_space.return_value = Space(s)

        self.space = Space(s)
        self.search = Random(self.mock_conn, s)

    def test_sample_all(self):
        db = list()
        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)

        param_set = set()

        for i, (token, params) in enumerate(iter(self.search.next, None)):
            params.update(token)
            param_set.add((params["a"], params["b"]))
            db.append(params)
            self.mock_conn.count_results.return_value = len(db)

        print(db)

        prod = list(self.space(p) for p in product(list(self.qa), list(self.qb)))
        prod_set = set((p["a"], p["b"]) for p in prod)

        print(prod)

        self.assertEqual(len(db), len(prod))
        self.assertEqual(param_set, prod_set)
