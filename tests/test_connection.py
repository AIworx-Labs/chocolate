import multiprocessing
import os
import pickle
import tempfile
import time
import unittest
import uuid

from hypothesis import given
from hypothesis.strategies import text

try:
    import pymongo
except ImportError:
    pymongo = None

from chocolate import SQLiteConnection, MongoDBConnection, DataFrameConnection, Space, uniform

if pymongo is not None:
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5)
    try:
        client.server_info()
    except pymongo.errors.ServerSelectionTimeoutError:
        mongodb = False
    else:
        mongodb = True


def lock_db(conn_class, *args):
    conn = conn_class(*args)
    with conn.lock():
        time.sleep(1)


class Base(object):
    def test_lock(self):
        p = multiprocessing.Process(target=lock_db, args=(self.conn_func,) + self.conn_args)
        p.start()

        timeout = False
        start_time = time.time()
        while time.time() - start_time < 10:
            try:
                with self.conn.lock(timeout=0.1):
                    pass
            except TimeoutError:
                timeout = True
                break

            time.sleep(0.1)

        p.join()
        self.assertEqual(timeout, True)

    def test_reentrant_lock(self):
        with self.conn.lock(timeout=1):
            with self.conn.lock(timeout=1):
                pass

    def test_results(self):
        data = [{"abc": 0, "def": 2}, {"abc": 1}, {"def": 42, "abc": 67, "hij": 23}]

        for d in data:
            self.conn.insert_result(d)

        res = self.conn.all_results()
        self.assertEqual(len(data), len(res))

        self.assertEqual(len(data), self.conn.count_results())

        res = sorted(res, key=lambda d: d["abc"])
        data = sorted(data, key=lambda d: d["abc"])

        for r, d in zip(res, data):
            for k, v in d.items():
                self.assertIn(k, r)
                self.assertEqual(v, r[k])

        res = self.conn.find_results({"abc": 67})
        self.assertEqual(1, len(res))
        self.assertIn("hij", res[0])
        self.assertIn("def", res[0])
        self.assertIn("abc", res[0])

    def test_update_result(self):
        data = {"abc": 0, "def": 2}
        token = {"_chocolate_id": 0}

        entry = data.copy()
        entry["loss"] = None
        entry.update(token)
        self.conn.insert_result(entry)

        values = {"_loss": 0.98}
        self.conn.update_result(token, values)

        res = self.conn.all_results()[0]
        self.assertEqual(values["_loss"], res["_loss"])

    def test_complementaries(self):
        data = [{"abc": 0, "def": 2}, {"abc": 1}, {"def": 42, "abc": 67}]

        for d in data:
            self.conn.insert_complementary(d)

        res = self.conn.all_complementary()
        self.assertEqual(len(data), len(res))

        res = sorted(res, key=lambda d: d["abc"])
        data = sorted(data, key=lambda d: d["abc"])

        for r, d in zip(res, data):
            for k, v in d.items():
                self.assertIn(k, r)
                self.assertEqual(v, r[k])

        res = self.conn.find_complementary(data[2])
        self.assertEqual(res["abc"], data[2]["abc"])

    def test_space(self):
        s = {"a": uniform(1, 2),
             "b": {"c": {"c1": uniform(0, 5)},
                   "d": {"d1": uniform(0, 6)}}}
        space = Space(s)

        space_read = self.conn.get_space()
        self.assertEqual(space_read, None)

        self.conn.insert_space(space)
        space_read = self.conn.get_space()

        self.assertEqual(space, space_read)
        self.assertRaises(AssertionError, self.conn.insert_space, space)

    def test_clear(self):
        self.conn.insert_result({"foo": "bar"})
        self.conn.insert_complementary({"bar": "spam", "foo": 2})
        self.conn.insert_space("some_data")

        self.conn.clear()
        self.assertEqual(self.conn.count_results(), 0)
        self.assertEqual(self.conn.all_complementary(), [])
        self.assertEqual(self.conn.get_space(), None)


class TestSQLite(unittest.TestCase, Base):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_name = "tmp.db"
        self.engine_str = "sqlite:///{}".format(os.path.join(self.tmp_dir.name, self.db_name))

        self.conn = SQLiteConnection(self.engine_str)

        self.conn_func = SQLiteConnection
        self.conn_args = (self.engine_str,)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_empty_name_connect(self):
        engine_str = "sqlite:///{}".format(os.path.join(self.tmp_dir.name, ""))
        self.assertRaises(RuntimeError, SQLiteConnection, engine_str)

    @given(text(alphabet="/ "))
    def test_invalid_ending_name_connect(self, s):
        engine_str = "sqlite:///{}".format(os.path.join(self.tmp_dir.name, s))
        self.assertRaises(RuntimeError, SQLiteConnection, engine_str)

    def test_no_uri_connect(self):
        engine_str = os.path.join(self.tmp_dir.name, self.db_name)
        self.assertRaises(RuntimeError, SQLiteConnection, engine_str)

    def test_memory_raises(self):
        engine_str = "sqlite:///:memory:"
        self.assertRaises(RuntimeError, SQLiteConnection, engine_str)


@unittest.skipIf(pymongo is None, "Cannot find pymongo module")
@unittest.skipIf(mongodb == False, "Cannot cannot connect to mongodb://localhost:27017/")
class TestMongoDB(unittest.TestCase, Base):
    def setUp(self):
        self.db_name = str(uuid.uuid1())
        self.engine_str = "mongodb://localhost:27017/"
        self.conn = MongoDBConnection(self.engine_str, database=self.db_name)

        self.conn_func = MongoDBConnection
        self.conn_args = (self.engine_str, self.db_name)

    def tearDown(self):
        self.conn.client.drop_database(self.db_name)


class TestDataFrame(unittest.TestCase, Base):
    def setUp(self):
        self.conn = DataFrameConnection()

    def test_lock(self):
        pass

    def test_pickle(self):
        data = [{"abc": 0, "def": 2}, {"abc": 1}, {"def": 42, "abc": 67, "hij": 23}]
        comp = [{"abc": 0, "def": 2}, {"abc": 1}, {"def": 42, "abc": 67, "hij": 23}]
        space = {"a": uniform(1, 2),
             "b": {"c": {"c1": uniform(0, 5)},
                   "d": {"d1": uniform(0, 6)}}}

        for d in data:
            self.conn.insert_result(d)

        for c in comp:
            self.conn.insert_complementary(c)

        self.conn.insert_space(Space(space))

        s = pickle.dumps(self.conn)
        l = pickle.loads(s)

        self.assertEqual(self.conn.results.equals(l.results), True)
        self.assertEqual(self.conn.complementary.equals(l.complementary), True)
        self.assertEqual(l.space, self.conn.space)
