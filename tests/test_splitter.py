from operator import itemgetter
import unittest
from unittest.mock import MagicMock

from chocolate.connection.splitter import split_space, transform_suboutput, ConnectionSplitter
from chocolate import Space, log, uniform, quantized_uniform


class TestSplitSpace(unittest.TestCase):
    def setUp(self):
        l1 = log(low=-3, high=5, base=10)
        l2 = log(low=-2, high=3, base=10)
        u = uniform(low=-1, high=1)
        qu = quantized_uniform(low=1, high=20, step=1)
        self.space = Space([{"algo": {"svm": {"C": l1,
                                              "kernel": {"linear": None,
                                                         "rbf": {"gamma": l2}},
                                              "cond2": {"aa": None,
                                                        "bb": {"abc": u}}},
                                      "knn": {"n_neighbors": qu}}},
                            {"cond3": 0, "p": l1, "p2": qu}])

    def test_names(self):
        subspaces_keys = set(k for s in split_space(self.space) for k in s([0] * len(s)).keys())
        self.assertSetEqual(set(self.space.names()), subspaces_keys)


class TestTransformSubOutput(unittest.TestCase):
    def setUp(self):
        l1 = log(low=-3, high=5, base=10)
        l2 = log(low=-2, high=3, base=10)
        u = uniform(low=-1, high=1)
        qu = quantized_uniform(low=1, high=20, step=1)
        self.space = Space([{"algo": {"svm": {"C": l1,
                                              "kernel": {"linear": None,
                                                         "rbf": {"gamma": l2}},
                                              "cond2": {"aa": None,
                                                        "bb": {"abc": u}}},
                                      "knn": {"n_neighbors": qu}}},
                            {"cond3": 0, "p": l1, "p2": qu}])

        self.params_keys = ["algo", "C", "kernel", "gamma", "cond2", "abc", "n_neighbors", "cond3", "p", "p2"]

    def test_output(self):
        subspaces = split_space(self.space)
        for s in subspaces:
            params = transform_suboutput(s([0] * len(s)), self.space)
            for key in params.keys():
                self.assertIn(key, self.params_keys)


class TestSplitter(unittest.TestCase):
    def setUp(self):
        self.mock_conn = MagicMock(name="connection")
        self.split_col = "_split_id"
        self.conn1 = ConnectionSplitter(self.mock_conn, 0, self.split_col)
        self.conn2 = ConnectionSplitter(self.mock_conn, 1, self.split_col)

    def test_lock(self):
        with self.conn1.lock():
            pass

        with self.conn2.lock():
            pass

        self.assertEqual(self.mock_conn.lock.call_count, 2)

    def test_all_results(self):
        db0 = [{"_chocolate_id": 0, "a": 0.1, "b": 0.0, "_loss": 6, self.split_col: 0},
               {"_chocolate_id": 1, "a": 0.2, "b": 0.0, "_loss": 7, self.split_col: 0}]
        db1 = [{"_chocolate_id": 0, "a": 0.3, "b": 0.0, "_loss": 8, self.split_col: 1}]

        self.mock_conn.all_results.return_value = db0 + db1

        res = self.conn1.all_results()
        self.assertEqual(len(res), len(db0))

        for r, d in zip(sorted(res, key=itemgetter("_chocolate_id")), sorted(db0, key=itemgetter("_chocolate_id"))):
            for k, v in d.items():
                self.assertIn(k, r)
                self.assertEqual(v, r[k])

        res = self.conn2.all_results()
        self.assertEqual(len(res), len(db1))

        for r, d in zip(sorted(res, key=itemgetter("_chocolate_id")), sorted(db1, key=itemgetter("_chocolate_id"))):
            for k, v in d.items():
                self.assertIn(k, r)
                self.assertEqual(v, r[k])

    def test_find_results(self):
        filter_ = {"_chocolate_id": 0}
        self.conn1.find_results(filter_)

        call_arg = filter_.copy()
        call_arg[self.split_col] = 0
        self.mock_conn.find_results.assert_called_with(call_arg)

        filter_ = {"_chocolate_id": 0}
        self.conn2.find_results(filter_)

        call_arg = filter_.copy()
        call_arg[self.split_col] = 1
        self.mock_conn.find_results.assert_called_with(call_arg)

    def test_insert_result(self):
        entry = {"_chocolate_id": 0, "a": 1}
        self.conn1.insert_result(entry)

        call_arg = entry.copy()
        call_arg[self.split_col] = 0
        self.mock_conn.insert_result.assert_called_with(call_arg)

        entry = {"_chocolate_id": 0, "a": 1}
        self.conn2.insert_result(entry)

        call_arg = entry.copy()
        call_arg[self.split_col] = 1
        self.mock_conn.insert_result.assert_called_with(call_arg)

    def test_update_result(self):
        entry = {"_chocolate_id": 0, "a": 1}
        value = 3
        self.conn1.update_result(entry, value)

        call_arg = entry.copy()
        call_arg[self.split_col] = 0
        self.mock_conn.update_result.assert_called_with(call_arg, value)

        entry = {"_chocolate_id": 0, "a": 1}
        value = 3
        self.conn2.update_result(entry, value)

        call_arg = entry.copy()
        call_arg[self.split_col] = 1
        self.mock_conn.update_result.assert_called_with(call_arg, value)

    def test_count_results(self):
        db0 = [{"_chocolate_id": 0, "a": 0.1, "b": 0.0, "_loss": 6, self.split_col: 0},
               {"_chocolate_id": 1, "a": 0.2, "b": 0.0, "_loss": 7, self.split_col: 0}]
        db1 = [{"_chocolate_id": 0, "a": 0.3, "b": 0.0, "_loss": 8, self.split_col: 1}]

        self.mock_conn.all_results.return_value = db0 + db1

        self.assertEqual(self.conn1.count_results(), 2)
        self.assertEqual(self.conn2.count_results(), 1)

    def test_all_complementary(self):
        db0 = [{"_chocolate_id": 0, "a": 0.1, "b": 0.0, self.split_col: 0},
               {"_chocolate_id": 1, "a": 0.2, "b": 0.0, self.split_col: 0}]
        db1 = [{"_chocolate_id": 0, "a": 0.3, "b": 0.0, self.split_col: 1}]

        self.mock_conn.all_complementary.return_value = db0 + db1

        res = self.conn1.all_complementary()
        self.assertEqual(len(res), len(db0))

        for r, d in zip(sorted(res, key=itemgetter("_chocolate_id")), sorted(db0, key=itemgetter("_chocolate_id"))):
            for k, v in d.items():
                self.assertIn(k, r)
                self.assertEqual(v, r[k])

        res = self.conn2.all_complementary()
        self.assertEqual(len(res), len(db1))

        for r, d in zip(sorted(res, key=itemgetter("_chocolate_id")), sorted(db1, key=itemgetter("_chocolate_id"))):
            for k, v in d.items():
                self.assertIn(k, r)
                self.assertEqual(v, r[k])

    def test_insert_complementary(self):
        entry = {"_chocolate_id": 0, "a": 1}
        self.conn1.insert_complementary(entry)

        call_arg = entry.copy()
        call_arg[self.split_col] = 0
        self.mock_conn.insert_complementary.assert_called_with(call_arg)

        entry = {"_chocolate_id": 0, "a": 1}
        self.conn2.insert_complementary(entry)

        call_arg = entry.copy()
        call_arg[self.split_col] = 1
        self.mock_conn.insert_complementary.assert_called_with(call_arg)

    def test_find_complementary(self):
        filter_ = {"_chocolate_id": 0}
        self.conn1.find_complementary(filter_)

        call_arg = filter_.copy()
        call_arg[self.split_col] = 0
        self.mock_conn.find_complementary.assert_called_with(call_arg)

        filter_ = {"_chocolate_id": 0}
        self.conn2.find_complementary(filter_)

        call_arg = filter_.copy()
        call_arg[self.split_col] = 1
        self.mock_conn.find_complementary.assert_called_with(call_arg)

    def test_get_space(self):
        s = {"a": uniform(1, 2),
             "b": {"c": {"c1": uniform(0, 5)},
                   "d": {"d1": uniform(0, 6)}}}
        space = Space(s)

        self.mock_conn.get_space.return_value = space

        self.assertEqual(space, self.conn1.get_space())
        self.assertEqual(space, self.conn2.get_space())

    def test_insert_space(self):
        s = {"a": uniform(1, 2),
             "b": {"c": {"c1": uniform(0, 5)},
                   "d": {"d1": uniform(0, 6)}}}
        space = Space(s)

        self.assertRaises(NotImplementedError, self.conn1.insert_space, space)
        self.assertRaises(NotImplementedError, self.conn2.insert_space, space)

    def test_clear(self):
        self.conn1.clear()
        self.mock_conn.clear.assert_called_once_with()
