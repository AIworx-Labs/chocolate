from collections import Mapping, Sequence, defaultdict

import numpy
import pandas

from .space import Space 


class Connection(object):
    """Abstract connection class that defines the database connection API.
    """
    def lock(self):
        raise NotImplementedError

    def all_results(self):
        raise NotImplementedError

    def find_results(self, filter):
        raise NotImplementedError

    def insert_result(self, entry):
        raise NotImplementedError

    def update_result(self, entry, value):
        raise NotImplementedError

    def count_results(self):
        raise NotImplementedError

    def all_complementary(self):
        raise NotImplementedError

    def insert_complementary(self, document):
        raise NotImplementedError

    def find_complementary(self, filter):
        raise NotImplementedError

    def get_space(self):
        raise NotImplementedError

    def insert_space(self, space):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def pop_id(self, document):
        raise NotImplementedError

    def results_as_dataframe(self):
        """Compile all the results and transform them using the space specified in the database. It is safe to
        use this method while other experiments are still writing to the database.

        Returns:
            A :class:`pandas.DataFrame` containing all results with its ``"_chocolate_id"`` as ``"id"``,
            their parameters and its loss. Pending results have a loss of :data:`None`.
        """
        with self.lock():
            s = self.get_space()
            results = self.all_results()

        all_results = []
        for r in results:
            #result_as_dict = {k: r[k] for k in s.names()}
            result = s([r[k] for k in s.names()])
            if "_loss" in r:
                result['loss'] = r['_loss']
            else:
                result['loss'] = None
            result["id"] = r["_chocolate_id"]
            all_results.append(result)

        df = pandas.DataFrame.from_dict(all_results)
        df.index = df.id
        df.drop("id", inplace=True, axis=1)
        return df


class RepeatCrossValidation(object):
    def __init__(self, repetitions, reduce=numpy.mean, rep_col="_repetition_id"):
        self.repetitions = repetitions
        self.reduce = reduce
        self.rep_col = rep_col
        self.space = None

    def _wrap_connection(self, connection):
        self.conn = connection
        self.orig_all_results = connection.all_results
        connection.all_results = self.all_results

        connection.count_results = self.count_results

    def all_results(self):
        results = self.orig_all_results()
        reduced_results = list()
        for result_group in self.group_repetitions(results):
            losses = [r["_loss"] for r in result_group if r["_loss"] is not None]
            if len(losses) > 0:
                result = result_group[0]
                result["_loss"] = self.reduce(losses)
                reduced_results.append(result)
            else:
                reduced_results.append(result_group[0])

        return reduced_results

    def count_results(self):
        return len(self.all_results())

    def next(self):
        """Has to be called inside a lock
        
        Returns:

        """
        if self.repetitions > 1:
            if self.space is None:
                self.space = self.conn.get_space()

            results = self.orig_all_results()
            names = set(self.space.names())
            names.add("_loss")
            for result_group in self.group_repetitions(results):
                if len(result_group) < self.repetitions:
                    vec = [result_group[0][k] if k in result_group[0] else None for k in self.space.names()]
                    token = {k: result_group[0][k] for k in result_group[0].keys() if k not in names}
                    token.update({self.rep_col: len(result_group)})
                    entry = result_group[0].copy()
                    # Ensure we don't have a duplicated id in the database
                    entry = self.conn.pop_id(entry)
                    token = self.conn.pop_id(token)
                    entry.update(token)
                    self.conn.insert_result(entry)
                    return token, self.space(vec)

            return {self.rep_col: 0}, None

        return None, None

    def group_repetitions(self, results):
        grouped = defaultdict(list)
        names = set(self.space.names())
        names.add("_loss")
        names.add(self.rep_col)

        for row in results:
            row = self.conn.pop_id(row)
            id_ = tuple((k, row[k]) for k in sorted(row.keys()) if k not in names)
            grouped[id_].append(row)

        return grouped.values()


class SearchAlgorithm(object):
    """Base class for search algorithms. Other than providing the :meth:`update` method
    it ensures the provided space fits with the one int the database.
    """
    def __init__(self, connection, space=None, crossvalidation=None, clear_db=False):
        if space is not None and not isinstance(space, Space):
            space = Space(space)

        self.conn = connection
        with self.conn.lock():
            db_space = self.conn.get_space()
            
            if space is None and db_space is None:
                raise RuntimeError("The database does not contain any space, please provide one through"
                    "the 'space' argument")
            
            elif space is not None and db_space is not None:
                if space != db_space and clear_db is False:
                    raise RuntimeError("The provided space and database space are different. To overwrite "
                        "the space contained in the database set the 'clear_db' argument")
                elif space != db_space and clear_db is True:
                    self.conn.clear()
                    self.conn.insert_space(space)

            elif space is not None and db_space is None:
                self.conn.insert_space(space)
            
            elif space is None and db_space is not None:
                space = db_space

        self.space = space

        self.crossvalidation = crossvalidation
        if self.crossvalidation is not None:
            self.crossvalidation = crossvalidation
            self.crossvalidation._wrap_connection(connection)

    def update(self, token, values):
        """Update the loss of the parameters associated with *token*.

        Args:
            token: A token generated by the sampling algorithm for the current
                parameters
            values: The loss of the current parameter set.

        """
        # Check and standardize values type
        if isinstance(values, Sequence):
            raise NotImplementedError("Cross-validation is not yet supported in DB")

        if isinstance(values, Sequence) and not isinstance(values[0], Mapping):
            raise NotImplementedError("Cross-validation is not yet supported in DB")
            values = [{"_loss" : v, "split_" : i} for i, v in enumerate(values)]
        elif not isinstance(values, Mapping):
            values = [{"_loss" : values}]
        elif isinstance(values, Mapping):
            values = [values]

        with self.conn.lock():
            if len(values) > 1:
                raise NotImplementedError("Cross-validation is not yet supported in DB")
                orig = self.conn.find_results(token)[0]
                orig = {k: orig[k] for k in self.space.column_names()}

            result = list()
            self.conn.update_result(token, values[0])
            for v in values[1:]:
                document = orig.copy()
                document.update(v)
                document.update(token)
                r = self.conn.insert_result(document)
                result.append(r)
            
        return result

    def next(self):
        """Retrieve the next point to evaluate based on available data in the
        database.
        
        Returns:
            A tuple containing a unique token and a fully qualified parameter set.
        """
        if self.crossvalidation is not None:
            reps_token, params = self.crossvalidation.next()
            if reps_token is not None and params is not None:
                return reps_token, params
            elif reps_token is not None and params is None:
                token, params = self._next(reps_token)
                return token, params

        return self._next()