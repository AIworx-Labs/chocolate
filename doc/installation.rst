Installation
============

Chocolate is installed using `pip <http://www.pip-installer.org/en/latest/>`_,
unfortunatelly we don't have any PyPI package yet. Here is the line you have to type ::

    pip install git+https://github.com/NovaSyst/chocolate@master

Dependencies
------------

Chocolate has various dependencies. While the optimizers depends on NumPy,
SciPy and Scikit-Learn, the SQLite database connection depends on dataset and 
filelock and the MongoDB database connection depends on PyMongo. Some utilities
depend on pandas. All but PyMongo will be installed with Chocolate.

