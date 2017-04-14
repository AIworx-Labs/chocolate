from collections import defaultdict

import numpy

class Repeat(object):
    def __init__(self, repetitions, reduce=numpy.mean, rep_col="_repetition_id"):
        self.repetitions = repetitions
        self.reduce = reduce
        self.rep_col = rep_col
        self.space = None

    def wrap_connection(self, connection):
        self.conn = connection
        self.orig_all_results = connection.all_results
        connection.all_results = self.all_results
        connection.count_results = self.count_results

    def all_results(self):
        results = self.orig_all_results()
        print(results)
        reduced_results = list()
        for result_group in self.group_repetitions(results):
            print(result_group)
            losses = [r["_loss"] for r in result_group if r["_loss"] is not None]
            if len(losses) > 0:
                result = result_group[0].copy()
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

        print(grouped)
        return grouped.values()
