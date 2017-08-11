Retrieving Results
==================

There is nothing easier than retrieving your results with Chocolate. Connections
define a method :meth:`~chocolate.SQLiteConnection.results_as_dataframe` that takes care of loading the data
from your database, transforming it back to your search space ranges and populating
a :class:`pandas.DataFrame`. This way you can use the powerful `pandas <http://pandas.pydata.org/>`_
and `seaborn <http://seaborn.pydata.org/>`_ libraries to analyse you results and not miss anything.
Here is how to get a nice pairwise plot of each parameter with the loss. ::

    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()

    from chocolate import SQLiteConnection

    conn = SQLiteConnection("sqlite:///chocolate.db")
    results = conn.results_as_dataframe()
    results = pd.melt(results, id_vars=["loss"], value_name='value', var_name="variable")

    sns.lmplot(x="value", y="loss", data=results, col="variable", col_wrap=3, sharex=False)

    plt.show()

And for those like me who are not patient enough to let the optimization finish,
the method :meth:`~chocolate.SQLiteConnection.results_as_dataframe` is multiprocess-safe
(thanks to our databases)!
