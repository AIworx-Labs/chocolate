import ghalton

from ..base import SearchAlgorithm


class QuasiRandom(SearchAlgorithm):
    """Quasi-Random sampler.

    Samples the search space using the generalized Halton low-discrepancy
    sequence. The underlying sequencer is the `ghalton
    <http://github.com/fmder/ghalton>`_ package, it must be installed
    separatly. The exploration is conducted independently of conditional
    search space, meaning that each subspace will receive approximately the
    same number of samples.

    This sampler will draw random numbers for each entry in the database to
    restore the random state for reproductibility when used concurrently with
    other random samplers. 

    Args:
        connection: A database connection object.
        space: The search space to explore with only discrete dimensions. The
            search space can be either a dictionary or a
            :class:`chocolate.Space` instance.
        clear_db: If set to :data:`True` and a conflict arise between the
            provided space and the space in the database, completely clear the
            database and insert set the space to the provided one.
        random_state: An integer used as seed to initialize the sequencer with or
            :data:`None` in which case the global state is used. This argument
            is ignored if :data:`permutations` if provided.
        permutations: Either, the string ``"ea"`` in which case the
            :data:`ghalton.EA_PERMS` are used or a valid list of permutations
            as desbribed in the ghalton package.
        skip: The number of points to skip in the sequence before the first
            point is sampled.

    """
    def __init__(self, connection, space, clear_db=False, random_state=None, permutations=None, skip=0):
        super(QuasiRandom, self).__init__(connection, space, clear_db)
        self.skip = skip
        if permutations == "ea":
            self.seq = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:len(self.space)])
        elif permutations is not None:
            self.seq = ghalton.GeneralizedHalton(permutations)
        elif random_state is not None:
            self.seq = ghalton.GeneralizedHalton(len(self.space), random_state)
        else:
            self.seq = ghalton.GeneralizedHalton(len(self.space))

        self.drawn = 0

    def next(self):
        """Retrieve the next quasi-random point to test and add it to the
        database with loss set to :data:`None`. On each call random points are
        burnt so that two random samplings running concurrently with the same
        random state don't draw the same points and produce reproductible
        results.

        Returns:
            A tuple containing a unique token and a fully qualified set of
            parameters.
        """
        with self.conn.lock():
            i = self.conn.count_results()
            token = {"_chocolate_id": i}

            # Burn the first i + skip points
            self.seq.get(self.skip + i - self.drawn)

            # Sample in [0, 1)^n
            out = self.seq.get(1)[0]
            self.drawn += i - self.drawn + 1

            # Signify next point to others using loss set to None
            # entry = {k : v for k, v in zip(self.space.names(), out)}
            entry = self.space(out, transform=False)
            # entry["_loss"] = None
            entry.update(token)
            self.conn.insert_result(entry)

        return token, self.space(out)
