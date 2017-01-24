
import multiprocessing
import os
import shutil
import time
import unittest
import uuid

try:
    import pymongo
except ImportError:
    pymongo = None

if pymongo is not None:
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5)
    try:
        client.server_info()
    except pymongo.errors.ServerSelectionTimeoutError:
        mongodb = False
    else:
        mongodb = True

from chocolate import SQLiteConnection, MongoDBConnection

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

    def test_results(self):
        data = [{"abc" : 0, "def" : 2}, {"abc" : 1}, {"def" : 42, "abc" : 67, "hij" : 23}]
        
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

        res = self.conn.find_results({"abc" : 67})
        self.assertEqual(1, len(res))
        self.assertIn("hij", res[0])
        self.assertIn("def", res[0])
        self.assertIn("abc", res[0])
        

    def test_update_result(self):
        data = {"abc" : 0, "def" : 2}
        token = {"_chocolate_id" : 0}

        entry = data.copy()
        entry["loss"] = None
        entry.update(token)
        self.conn.insert_result(entry)
        
        values = {"_loss" : 0.98}
        self.conn.update_result(token, values)

        res = self.conn.all_results()[0]
        self.assertEqual(values["_loss"], res["_loss"])

    def test_complementaries(self):
        data = [{"abc" : 0, "def" : 2}, {"abc" : 1}, {"def" : 42, "abc" : 67}]

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


class TestSQLite(unittest.TestCase, Base):
    def setUp(self):
        self.db_path = str(uuid.uuid1())
        self.db_name = "tmp.db"
        self.engine_str = "sqlite:///{}".format(os.path.join(self.db_path, self.db_name))

        os.mkdir(self.db_path)
        self.conn = SQLiteConnection(self.engine_str)

        self.conn_func = SQLiteConnection
        self.conn_args = (self.engine_str,)

    def tearDown(self):
        shutil.rmtree(self.db_path)

@unittest.skipIf(pymongo is None, "Cannot find pymongo module")
@unittest.skipIf(mongodb == False, "Cannot cannot connect to mongodb://localhost:27017/")
class TestMongoDB(unittest.TestCase, Base):
    def setUp(self):
        self.db_name = str(uuid.uuid1())
        self.conn = MongoDBConnection("mongodb://localhost:27017/", database=self.db_name)

        self.conn_func = MongoDBConnection
        self.conn_args = ("mongodb://localhost:27017/", self.db_name)

    def tearDown(self):
        self.conn.client.drop_database(self.db_name)