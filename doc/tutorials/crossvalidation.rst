Cross-validating Optimization
=============================

More often than not, the optimized process results have some variability. To make the
optimization process more robust each parameter set has to be evaluated more than once.
Chocolate provides seemless cross-validation in the search algorithms. The cross-validation
object, if provided, intercepts calls to the database and ensures every experiment is
repeated a given number of times. Cross-validations, just like every other experiments, is
done in parallel and asynchroneously. To use cross-validation simply create a
cross-validation object and assign it to the search algorithm. ::

    import numpy as np
    import chocolate as choco

    def evaluate(p1, p2):
        return p1 + p2 + np.random.randn()

    if __name__ == "__main__":
        space = {"p1": choco.uniform(0, 10), "p2": choco.uniform(0, 5)}
        connection = choco.SQLiteConnection(url="sqlite:///cv.db")
        cv = choco.Repeat(repetitions=3, reduce=np.mean, rep_col="_repetition_id")
        s = choco.Grid(space, connection, crossvalidation=cv)

        token, params = s.next()
        loss = evaluate(**params)
        print(token, params, loss)
        s.update(token, loss)

The preceding script, if run a couple of times, will output the following tokens and parameters
(with probably different parameters). ::

    {'_repetition_id': 0, '_chocolate_id': 0} {'p1': 8.1935000833291518, 'p2': 4.2668676560356529} 13.886112047266854
    {'_repetition_id': 1, '_chocolate_id': 0} {'p1': 8.1935000833291518, 'p2': 4.2668676560356529} 11.394347119228563
    {'_repetition_id': 2, '_chocolate_id': 0} {'p1': 8.1935000833291518, 'p2': 4.2668676560356529} 10.790294230308477
    {'_repetition_id': 0, '_chocolate_id': 1} {'p1': 7.4031022047092732, 'p2': 0.14633280691567885} 6.349087103521951
    {'_repetition_id': 1, '_chocolate_id': 1} {'p1': 7.4031022047092732, 'p2': 0.14633280691567885} 6.269733948749414
    {'_repetition_id': 2, '_chocolate_id': 1} {'p1': 7.4031022047092732, 'p2': 0.14633280691567885} 6.895059981273982
    {'_repetition_id': 0, '_chocolate_id': 2} {'p1': 2.4955760398088778, 'p2': 4.4722460515061} 6.82570693646037

.. note::

   The cross-validation is not responsible of shuffling your dataset. You must include
   this step in your script.

The cross-validation object wraps the connection to reduce the loss of experiments with same
``"_chocolate_id"``. Thus, algorithms never see the repetitions, they only receive a single
parameter set with the reduced loss. For the last example, the algorithms,
when interrogating the database, will see the following parameter sets and losses. ::

    {'p1': 8.1935000833291518, 'p2': 4.2668676560356529} 12.023584465601298
    {'p1': 7.4031022047092732, 'p2': 0.14633280691567885} 6.5046270111817819
    {'p1': 2.4955760398088778, 'p2': 4.4722460515061} 6.82570693646037
