Basics
------

Let strat with the very basics. Suppose you want to optimize the parameters of the himmelblau function
:math:`f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2` with Chocolate ::

    def himmelblau(x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

You'd first have to define a :ref:`search space <Search Space Representation>` 
for parameters x and y ::

    import chocolate as choco

    space = {"x" : choco.uniform(-6, 6),
             "y" : choco.uniform(-6, 6)}

Next, you'd establish where the results should be saved. We have two database
adaptors one :class:`~chocolate.SQLiteConnection` (which we prefer) and one
:class:`~chocolate.MongoDBConnection` (which we also like, of course!) ::

    conn = choco.SQLiteConnection("sqlite:///my_db.db")

.. note:: While the SQLite adaptor should be used when a common file system is
   available for all compute nodes, the MongoDB adaptor is more suited
   when compute nodes cannot share such file system (i.e. Amazon EC2 spot
   instances)

When this overwhelming task is done, you'd choose from our :ref:`sampling
<Sampling Algorithms>` or :ref:`search
<Search Algorithms>` algorithms the one that you prefer. We will use quasi
random sampling because its ze best. ::

    sampler = choco.QuasiRandom(conn, space, random_state=42, skip=0)

Now comes the funny part, using Chocolate, a process usually does one and only
one evaluation. So you'd just have to ask the sampler: "Hey ya! What's the
next point I should evaluate yo?", do the evaluation an tell the sampler about
it. ::

    token, params = sampler.next()
    loss = himmelblau(**params)
    sampler.update(token, loss)

Ho yeah, the token is just a unique id we use to trace the parameters in the
database. You can sure have a look at it. ::

    >>> print(token)
    {"_chocolate_id" : 0}