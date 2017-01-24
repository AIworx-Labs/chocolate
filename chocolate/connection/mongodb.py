
from contextlib import contextmanager
import re
import time

from pymongo import MongoClient

from ..base import Connection

class MongoDBConnection(Connection):
    """Connection to a MongoDB database.

    Args:
        url: Full url to the database including credentials but omiting the
            database and the collection.
        database: The database in the MongoDB engine.
        result_col: Collection used to store the experiences and their
            results.
        complementary_col: Collection used to store complementary information
            necessary to the optimizer.
    """
    def __init__(self, url, database="chocolate", result_col="results", complementary_col="complementary"):
        if not MongoClient:
            raise 
        
        self.client = MongoClient(url)
        self.db = self.client[database]
        self.results = self.db[result_col]
        self.complementary = self.db[complementary_col]
        self._lock = self.db.lock

    @contextmanager
    def lock(self, timeout=-1, poll_interval=0.05):
        """Context manager that locks the entire database.
        
        ::
        
            conn = MongoDBConnection("mongodb://localhost:27017/")
            with conn.lock(timeout=5):
                # The database is lock
                all_ = conn.all_results()
                conn.insert({"new_data" : len(all_)})

            # The database is unlocked

        Args:
            timeout: If the lock could not be acquired in *timeout* seconds
                raises a timeout error. If 0 or less, wait forever.
            poll_interval: Number of seconds between lock acquisition tryouts.

        Raises:
            TimeoutError: Raised if the lock could not be acquired.
        """
        start_time = time.time()
        l = self._lock.find_one_and_update({"name" : "lock"},
                                           {"$set" : {"lock" : True}},
                                           upsert=True)

        while l is not None and l["lock"] != False and timeout != 0:
            time.sleep(poll_interval)
            l = self._lock.find_one_and_update({"name" : "lock"},
                                               {"$set" : {"lock" : True}},
                                               upsert=True)

            if time.time() - start_time > timeout:
                break

        if l is None or l["lock"] == False:
            # The lock is acquired
            try:
                yield
            finally:
                l = self._lock.find_one_and_update({"name" : "lock"},
                                                   {"$set" : {"lock" : False}})
        else:
            raise TimeoutError("Could not acquire MongoDB lock")

    def all_results(self):
        """Get all entries of the result table as a list. The order is
        undefined.
        """
        return list(self.results.find())

    def insert_result(self, document):
        """Insert a new *document* in the result table.
        """
        return self.results.insert_one(document)

    def update_result(self, token, values):
        """Update or add *values* of a given document in the result table.

        Args:
            token: A unique identifier of the document to update.
            value: A mapping of values to update or add.
        """
        return self.results.update_one(token, {"$set" : values})

    def count_results(self):
        """Get the total number of entries in the result table.
        """
        return self.results.count()

    def all_complementary(self):
        """Get all entries of the complementary information table as a list.
        The order is undefined.
        """
        return list(self.complementary.find())

    def insert_complementary(self, document):
        """Insert a new document in the complementary information table.
        """
        return self.complementary.insert_one(document)

    def find_complementary(self, filter):
        """Find a document from the complementary information table.
        """
        return self.complementary.find_one(filter)