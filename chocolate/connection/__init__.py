
try:
    import pymongo
except ImportError:
    pass
else:
    from .mongodb import MongoDBConnection
    
from .sqlite import SQLiteConnection