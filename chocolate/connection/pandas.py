from contextlib import contextmanager

import numpy
import pandas

from ..base import Connection


class DataFrameConnection(Connection):
    """Connection to a pandas DataFrame.

    This connection is absolutely not meant to be used in concurrent processes.
    In fact, using this connection in different processes will result in two
    independent searches **not** sharing any information.
    """
    def __init__(self):
        self.results = pandas.DataFrame()
        self.complementary = pandas.DataFrame()
        self.space = None

    @contextmanager
    def lock(self, *args, **kwargs):
        """This function does not lock anything. Do not use in concurrent
        processes.
        """
        yield

    def all_results(self):
        """Get a list of all entries of the result table. The order is
        undefined.
        """
        return list(self.results.T.to_dict().values())

    def find_results(self, filter):
        """Get a list of all results associated with *filter*. The order is
        undefined.
        """
        selection = self.results
        for k, v in filter.items():
            selection = selection[selection[k] == v]
        return list(selection.T.to_dict().values())

    def insert_result(self, document):
        """Insert a new *document* in the result data frame. The columns does
        not need to be defined nor all present. Any new column will be added
        to the database and any missing column will get value None.
        """
        self.results = self.results.append(document, ignore_index=True)

    def update_result(self, document, value):
        """Update or add *values* of a given row in the result data frame.

        Args:
            filter: A unique identifier of the row to update.
            values: A mapping of values to update or add.
        """
        size = len(self.results.index)
        selection = [True] * size
        for k, v in document.items():
            selection = numpy.logical_and(self.results[k] == v, selection)

        for k, v in value.items():
            if not k in self.results:
                self.results[k] = pandas.Series([None] * size)
            self.results.loc[selection, k] = v

    def count_results(self):
        """Get the total number of entries in the result table."""
        return len(self.results.index)

    def all_complementary(self):
        """Get all entries of the complementary information table as a list.
        The order is undefined.
        """
        return list(self.complementary.T.to_dict().values())

    def insert_complementary(self, document):
        """Insert a new document (row) in the complementary information data frame."""
        self.complementary = self.complementary.append(document, ignore_index=True)

    def find_complementary(self, filter):
        """Find a document (row) from the complementary information data frame."""
        selection = self.complementary
        for k, v in filter.items():
            selection = selection[selection[k] == v]
        return list(selection.T.to_dict().values())[0]

    def get_space(self):
        """Returns the space used for previous experiments."""
        return self.space

    def insert_space(self, space):
        """Insert a space in the database.

        Raises:
            AssertionError: If a space is already present.
        """
        assert self.space is None, "Space table cannot contain more than one space, clear table first."
        self.space = space

    def clear(self):
        """Clear all data."""
        self.results = pandas.DataFrame()
        self.complementary = pandas.DataFrame()
        self.space = None
