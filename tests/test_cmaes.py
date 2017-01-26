
import unittest
from unittest.mock import MagicMock

from chocolate import CMAES
from chocolate.space import *

class TestCMAES(unittest.TestCase):
    def setUp(self):
        s = {"a" : uniform(1, 10),
             "b" : uniform(5, 15)}

        self.mock_conn = MagicMock(name="connection")
        self.mock_conn.get_space.return_value = Space(s)
        
        self.search = CMAES(self.mock_conn, s)

    def test_cold_start(self):
        self.mock_conn.all_results.return_value = []
        self.mock_conn.count_results.return_value = 0
        self.mock_conn.all_complementary.return_value = []

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 0)

        self.assertIn("a", p)

    def test_bootstrap_low(self):
        db = [{"_chocolate_id" : 0, "a" : 0, "b" : 0, "_loss" : 0.1},
              {"_chocolate_id" : 1, "a" : 0.5, "b" : 0.5, "_loss" : 0.1}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = []

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 2)

        self.assertIn("a", p)

    def test_bootstrap_high(self):
        db = [{"_chocolate_id" : 0, "a" : 0, "b" : 0, "_loss" : 0.1},
              {"_chocolate_id" : 1, "a" : 0.5, "b" : 0.5, "_loss" : 0.2},
              {"_chocolate_id" : 2, "a" : 0.0, "b" : 0.5, "_loss" : 0.3},
              {"_chocolate_id" : 3, "a" : 0.5, "b" : 0.0, "_loss" : 0.4},
              {"_chocolate_id" : 4, "a" : 0.9, "b" : 0.0, "_loss" : 0.5}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = []

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 5)

        self.assertIn("a", p)

    def test_bootstrap_all_none(self):
        db = [{"_chocolate_id" : 0, "a" : 0, "b" : 0, "_loss" : None},
              {"_chocolate_id" : 1, "a" : 0.5, "b" : 0.5, "_loss" : None},
              {"_chocolate_id" : 2, "a" : 0.0, "b" : 0.5, "_loss" : None},
              {"_chocolate_id" : 3, "a" : 0.5, "b" : 0.0, "_loss" : None},
              {"_chocolate_id" : 4, "a" : 0.9, "b" : 0.0, "_loss" : None}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = []

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 5)

        self.assertIn("a", p)

    def test_bootstrap_complementary(self):
        db = [{"_chocolate_id" : 0, "a" : 0, "b" : 0, "_loss" : 1},
              {"_chocolate_id" : 1, "a" : 0.5, "b" : 0.5, "_loss" : 2},
              {"_chocolate_id" : 2, "a" : 0.0, "b" : 0.5, "_loss" : 3},
              {"_chocolate_id" : 3, "a" : 0.5, "b" : 0.0, "_loss" : 4},
              {"_chocolate_id" : 4, "a" : 0.9, "b" : 0.0, "_loss" : 5},
              {"_chocolate_id" : 5, "a" : 0.9, "b" : 0.0, "_loss" : 6},
              {"_chocolate_id" : 6, "a" : 0.9, "b" : 0.0, "_loss" : 7},
              {"_chocolate_id" : 7, "a" : 0.9, "b" : 0.0, "_loss" : 8}]

        comp = [{"_chocolate_id" : 5, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 0},
                {"_chocolate_id" : 6, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 0},
                {"_chocolate_id" : 7, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 0}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = comp

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 8)

        self.assertIn("a", p)

    def test_warm_start(self):
        db = [{"_chocolate_id" : 0, "a" : 0.9, "b" : 0.0, "_loss" : 6},
              {"_chocolate_id" : 1, "a" : 0.9, "b" : 0.0, "_loss" : 7},
              {"_chocolate_id" : 2, "a" : 0.9, "b" : 0.0, "_loss" : 8}]

        comp = [{"_chocolate_id" : 0, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 0},
                {"_chocolate_id" : 1, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 0},
                {"_chocolate_id" : 2, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 0}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = comp

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 3)

        self.assertIn("a", p)

    def test_invalid_candidate(self):
        db = [{"_chocolate_id" : 0, "a" : 0.9, "b" : 0.0, "_loss" : 6},
              {"_chocolate_id" : 1, "a" : 0.9, "b" : 0.0, "_loss" : 7},
              {"_chocolate_id" : 2, "a" : 0.9, "b" : 0.0, "_loss" : 0}]

        comp = [{"_chocolate_id" : 0, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 0},
                {"_chocolate_id" : 1, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 1},
                {"_chocolate_id" : 1, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 0},
                {"_chocolate_id" : 2, "a" : 0.9, "b" : 0.0, "_ancestor_id" : 0, "_invalid" : 0}]

        self.mock_conn.all_results.return_value = db
        self.mock_conn.count_results.return_value = len(db)
        self.mock_conn.all_complementary.return_value = comp

        token, p = self.search.next()

        self.assertIn("_chocolate_id", token)
        self.assertEqual(token["_chocolate_id"], 3)

        self.assertIn("a", p)