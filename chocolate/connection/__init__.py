
try:
    import pymongo
else:
    from .mongodb import MongoDBConnection
    
from .sqlite import SQLiteConnection