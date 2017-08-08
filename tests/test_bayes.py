import unittest
from unittest.mock import MagicMock

import numpy
from chocolate.space import *

from chocolate import Bayes


class TestBayes(unittest.TestCase):
    def setUp(self):
        s = {
            "a": uniform(1, 10),
            "b": uniform(5, 15)
        }

        self.mock_conn = MagicMock(name="connection")
        self.mock_conn.get_space.return_value = Space(s)

        self.search = Bayes(self.mock_conn, s)

    def test_cold_start(self):
        self.mock_conn.all_results.return_value = []
        self.mock_conn.count_results.return_value = 0
        self.mock_conn.all_complementary.return_value = []

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 0)

        self.assertIn("a", p)

    def test_bootstrap_low(self):
        db = [{"_chocolate_id": 0, "a": 0, "b": 0, "_loss": 0.1},
              {"_chocolate_id": 1, "a": 0.5, "b": 0.5, "_loss": 0.1}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = []

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 2)

        self.assertIn("a", p)

    def test_bootstrap_high(self):
        db = [{"_chocolate_id": i, "a": i * (1.0 / 30), "b": i * (1.0 / 30), "_loss": i * (1.0 / 30)} for i in
              range(30)]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = []

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 30)

        self.assertIn("a", p)

    def test_bootstrap_all_none(self):
        db = [{"_chocolate_id": 0, "a": 0, "b": 0, "_loss": None},
              {"_chocolate_id": 1, "a": 0.5, "b": 0.5, "_loss": None},
              {"_chocolate_id": 2, "a": 0.0, "b": 0.5, "_loss": None},
              {"_chocolate_id": 3, "a": 0.5, "b": 0.0, "_loss": None},
              {"_chocolate_id": 4, "a": 0.9, "b": 0.0, "_loss": None}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = []

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 5)

        self.assertIn("a", p)

    def test_warm_start(self):
        db = [{"_chocolate_id": 0, "a": 0.9, "b": 0.0, "_loss": 6},
              {"_chocolate_id": 1, "a": 0.9, "b": 0.0, "_loss": 7},
              {"_chocolate_id": 2, "a": 0.9, "b": 0.0, "_loss": 8}]

        comp = [{"_chocolate_id": 0, "a": 0.9, "b": 0.0, "_ancestor_id": 0, "_invalid": 0},
                {"_chocolate_id": 1, "a": 0.9, "b": 0.0, "_ancestor_id": 0, "_invalid": 0},
                {"_chocolate_id": 2, "a": 0.9, "b": 0.0, "_ancestor_id": 0, "_invalid": 0}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = comp

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 3)

        self.assertIn("a", p)

    def test_invalid_candidate(self):
        db = [{"_chocolate_id": 0, "a": 0.9, "b": 0.0, "_loss": 6},
              {"_chocolate_id": 1, "a": 0.9, "b": 0.0, "_loss": 7},
              {"_chocolate_id": 2, "a": 0.9, "b": 0.0, "_loss": 0}]

        comp = [{"_chocolate_id": 0, "a": 0.9, "b": 0.0, "_ancestor_id": 0, "_invalid": 0},
                {"_chocolate_id": 1, "a": 0.9, "b": 0.0, "_ancestor_id": 0, "_invalid": 1},
                {"_chocolate_id": 1, "a": 0.9, "b": 0.0, "_ancestor_id": 0, "_invalid": 0},
                {"_chocolate_id": 2, "a": 0.9, "b": 0.0, "_ancestor_id": 0, "_invalid": 0}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = comp

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 3)

        self.assertIn("a", p)

    def test_ei(self):
        db = self._create_fake_db()
        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = []

        gp, y = self.search._fit_gp(*self.search._load_database())
        out = self.search._ei([0.25, 0.25], gp, y.max(), 0.1)

        self.assertIsNotNone(out)

    def test_ucb(self):
        db = self._create_fake_db()
        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = []

        gp, y = self.search._fit_gp(*self.search._load_database())
        out = self.search._ucb([0.25, 0.25], gp, 2.675)

        self.assertIsNotNone(out)

    def test_acquisition_func(self):
    #     #db = [{"_chocolate_id" : 0, "a" : 0, "b" : 0, "_loss" : 0.1},
    #     #     {"_chocolate_id" : 1, "a" : 0.5, "b" : 0.5, "_loss" : 0.1}]
        db = self._create_fake_db()
        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = []
        gp, y = self.search._fit_gp(*self.search._load_database())
        out = self.search._acquisition(gp, y)
        self.assertIsNotNone(out)

    def test_conditional_n_steps(self):
        s = [{"a": uniform(1, 10), "b": uniform(5, 15), "C": 0},
             {"c": uniform(2, 3), "C": 1}]

        db = list()

        self.mock_conn = MagicMock(name="connection")
        self.mock_conn.get_space.return_value = Space(s)
        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = 0

        self.search = Bayes(self.mock_conn, s)

        for i in range(15):
            token, p = self.search.next()

            self.assertIn("_chocolate_id", token)
            self.assertEqual(token["_chocolate_id"], i)

            entry = self.mock_conn.insert_result.call_args[0][0]
            entry["_loss"] = numpy.random.randn()

            db.append(entry)
            self.mock_conn.count_results.return_value = len(db)

    def _create_fake_db(self):
        db = [{
            "_chocolate_id": i,
            "a":             numpy.random.random(),
            "b":             numpy.random.random(),
            "_loss":         numpy.random.random()
        } for i in range(25)]
        return db
