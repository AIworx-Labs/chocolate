# Chocolate
Chocolate is a **completely asynchronous** optimisation framework relying solely on a
database to share information between workers. Chocolate uses **no master process** for
distributing tasks. Every task is completely independent and only gets its
information from a database. Chocolate is thus ideal in controlled computing
environments where it is hard to maintain a master process for the duration
of the optimisation.

Chocolate has been designed and optimized for hyperparameter optimization where
each function evaluation takes very long to complete and is difficult to parallelize.
Chocolate allows optimization over **conditional search spaces** either as using
conditional kernels in a Bayesian optimizer or as a multi-armed bandit problem using
Thompson sampling. Chocolate also handles **multi-objective optimisation** where
multiple loss funtions are optimized simultaneously.

Chocolate provides the following sampling/searching algorithms:
- Grid
- Random
- QuasiRandom
- CMAES
- MOCMAES
- Bayesian

Chocolat is licensed under the [3-Clause BSD License](http://opensource.org/licenses/BSD-3-Clause)

## Documentation

The full documentation is available at
http://chocolate.readthedocs.io.

## Installation

Chocolate is installed using [pip](http://www.pip-installer.org/en/latest/),
unfortunately we don't have any PyPI package yet. Here is the line you have to type

`pip install git+https://github.com/NovaSyst/chocolate@master`

## Dependencies

Chocolate has various dependencies. While the optimizers depends on NumPy,
SciPy and Scikit-Learn, the SQLite database connection depends on dataset and 
filelock and the MongoDB database connection depends on PyMongo. Some utilities
depend on pandas. All but PyMongo will be installed with Chocolate.

## Simple example

The following very simple example shows how to optimize a conditional search space in
Chocolate. You'll note that a single point is sampled and evaluated in the script. Since
the database connections are 'parallel' safe, you can run this script in concurrent processes
and achieve maximum parallelism. 

```python
import chocolate as choco

def objective_function(condition, x=None, y=None):
    """An objective function returning ``1 - x`` when *condition* is 1 and 
    ``y - 6`` when *condition* is 2.
    
    Raises:
        ValueError: If condition is different than 1 or 2.
    """
    if condition == 1:
        return 1 - x
    elif condition == 2:
        return y - 6
    raise ValueError("condition must be 1 or 2, got {}.".format(condition))

# Define the conditional search space 
space = [
            {"condition": 1, "x": choco.uniform(low=1, high=10)},
            {"condition": 2, "y": choco.log(low=-2, high=2, base=10)}
        ]

# Establish a connection to a SQLite local database
conn = choco.SQLiteConnection("sqlite:///my_db.db")

# Construct the optimizer
sampler = choco.Bayes(conn, space)

# Sample the next point
token, params = sampler.next()

# Calculate the loss for the sampled point (minimized)
loss = objective_function(**params)

# Add the loss to the database
sampler.update(token, loss)
```

Have a look at the [documentation](http://chocolate.readthedocs.io) tutorials for more examples.
