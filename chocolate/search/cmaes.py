from itertools import groupby
from operator import itemgetter

import numpy

from ..base import SearchAlgorithm


# TODO: Use self random state
class CMAES(SearchAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy minimization method.

    A CMA-ES strategy that combines the :math:`(1 + \\lambda)` paradigm
    [Igel2007]_, the mixed integer modification [Hansen2011]_, active
    covariance update [Arnold2010]_ and covariance update for constrained
    optimization [Arnold2012]_. It generates a single new point per
    iteration and adds a random step mutation to dimensions that undergoes a
    too small modification. Even if it includes the mixed integer
    modification, CMA-ES does not handle well dimensions without variance and
    thus it should be used with care on search spaces with conditional
    dimensions.

    Args:
        connection: A database connection object.
        space: The search space to explore.
        clear_db: If set to :data:`True` and a conflict arise between the
            provided space and the space in the database, completely clear the
            database and set the space to the provided one.
        random_state: An instance of :class:`~numpy.random.RandomState`, an
            object to initialize the internal random state with, or None, in
            which case the global numpy random state is used.
        **params: Additional parameters to pass to the strategy as described in
            the following table, along with default values.

            +----------------+---------------------------+----------------------------+
            | Parameter      | Default value             | Details                    |
            +================+===========================+============================+
            | ``d``          | ``1 + ndim / 2``          | Damping for step-size.     |
            +----------------+---------------------------+----------------------------+
            | ``ptarg``      | ``1 / 3``                 | Taget success rate.        |
            +----------------+---------------------------+----------------------------+
            | ``cp``         | ``ptarg / (2 + ptarg)``   | Step size learning rate.   |
            +----------------+---------------------------+----------------------------+
            | ``cc``         | ``2 / (ndim + 2)``        | Cumulation time horizon.   |
            +----------------+---------------------------+----------------------------+
            | ``ccovp``      | ``2 / (ndim**2 + 6)``     | Covariance matrix positive |
            |                |                           | learning rate.             |
            +----------------+---------------------------+----------------------------+
            | ``ccovn``      | ``0.4 / (ndim**1.6 + 1)`` | Covariance matrix negative |
            |                |                           | learning rate.             |
            +----------------+---------------------------+----------------------------+
            | ``beta``       | ``0.1 / (ndim + 2)``      | Covariance matrix          |
            |                |                           | constraint learning rate.  |
            +----------------+---------------------------+----------------------------+
            | ``pthresh``    | ``0.44``                  | Threshold success rate.    |
            +----------------+---------------------------+----------------------------+

    .. [Igel2007] Igel, Hansen, Roth. Covariance matrix adaptation for
       multi-objective optimization. 2007

    .. [Arnold2010] Arnold and Hansen. Active covariance matrix adaptation for
       the (1 + 1)-CMA-ES. 2010.

    .. [Hansen2011] Hansen. A CMA-ES for Mixed-Integer Nonlinear Optimization.
       Research Report] RR-7751, INRIA. 2011

    .. [Arnold2012] Arnold and Hansen. A (1 + 1)-CMA-ES for Constrained
        Optimisation. 2012
    """

    def __init__(self, connection, space, clear_db=False, random_state=None, **params):
        super(CMAES, self).__init__(connection, space, clear_db)
        self.params = params

        if isinstance(random_state, numpy.random.RandomState):
            self.random_state = random_state
        elif random_state is None:
            self.random_state = numpy.random
        else:
            self.random_state = numpy.random.RandomState(random_state)

    def next(self):
        """Retrieve the next point to evaluate based on available data in the
        database. Each time :meth:`next` is called, the algorithm will reinitialize
        it-self based on the data in the database.

        Returns:
            A tuple containing a unique token and a fully qualified parameter set.
        """
        self._init()

        with self.conn.lock():
            # Check what is available in that database
            results = {r["_chocolate_id"]: r for r in self.conn.all_results()}
            ancestors, ancestors_ids = self._load_ancestors(results)
            bootstrap = self._load_bootstrap(results, ancestors_ids)

            # Rank-mu update on individuals created from another algorithm
            self._bootstrap(bootstrap)
            # _bootstrab sets the parent if enough candidates are available (>= 4)

            # If the parent is still None and ancestors are available
            # set the parent to the first evaluated candidate if any
            if self.parent is None and len(ancestors) > 0:
                self.parent = next((a for a in ancestors if a["loss"] is not None), None)

            # Generate the next point
            token = {"_chocolate_id": self.conn.count_results()}

            # If the parent is still None, no information available
            if self.parent is None:
                # out = numpy.ones(self.dim) / 2.0
                out = numpy.random.rand(self.dim)

                # Signify the first point to others using loss set to None
                # Transform to dict with parameter names
                # entry = {str(k): v for k, v in zip(self.space.names(), out)}
                entry = self.space(out, transform=False)
                # entry["_loss"] = None
                entry.update(token)
                self.conn.insert_result(entry)

                # Add the step to the complementary table
                # Transform to dict with parameter names
                # entry = {str(k): v for k, v in zip(self.space.names(), out)}
                entry = self.space(out, transform=False)
                entry.update(_ancestor_id=-1, _invalid=0, _search_algo="cmaes", **token)
                self.conn.insert_complementary(entry)

                # return the true parameter set
                return token, self.space(out)

            else:
                # Simulate the CMA-ES update for each ancestor.
                for key, group in groupby(ancestors[1:], key=itemgetter("ancestor_id")):
                    # If the loss for this entry is not yet availabe, don't include it
                    group = list(group)
                    self.lambda_ = len(group)
                    self._configure()  # Adjust constants that depends on lambda
                    self._update_internals(group)

                invalid = 1
                while invalid > 0:
                    # Generate a single candidate at a time
                    self.lambda_ = 1
                    self._configure()

                    # The ancestor id is the last candidate that participated in the
                    # covariance matrix update
                    ancestor_id = next(
                        (a["chocolate_id"] for a in reversed(bootstrap + ancestors) if a["loss"] is not None or a[
                            "invalid"] > 0),
                        None)
                    assert ancestor_id is not None, "Invalid ancestor id"

                    out, y = self._generate()

                    # Encode constraint violation
                    invalid = sum(2 ** (2 * i) for i, xi in enumerate(out) if xi < 0)
                    invalid += sum(2 ** (2 * i + 1) for i, xi in enumerate(out) if xi >= 1)

                    # Add the step to the complementary table
                    # Transform to dict with parameter names
                    # entry = {str(k): v for k, v in zip(self.space.names(), y)}
                    entry = self.space(y, transform=False)
                    entry.update(_ancestor_id=ancestor_id, _invalid=invalid, _search_algo="cmaes", **token)
                    self.conn.insert_complementary(entry)

                # Signify next point to others using loss set to None
                # Transform to dict with parameter names
                # entry = {str(k): v for k, v in zip(self.space.names(), out)}
                entry = self.space(out, transform=False)
                # entry["_loss"] = None
                entry.update(token)
                self.conn.insert_result(entry)

                # return the true parameter set
                return token, self.space(out)

    def _init(self):
        self.parent = None
        self.sigma = 0.2
        self.dim = len(self.space)

        self.C = numpy.identity(self.dim)
        self.A = numpy.linalg.cholesky(self.C)

        self.pc = numpy.zeros(self.dim)

        # Covariance matrix adaptation
        self.cc = self.params.get("cc", 2.0 / (self.dim + 2.0))
        self.ccovp = self.params.get("ccovp", 2.0 / (self.dim ** 2 + 6.0))
        self.ccovn = self.params.get("ccovn", 0.4 / (self.dim ** 1.6 + 1.0))
        self.beta = self.params.get("beta", 0.1 / (self.dim + 2.0))
        self.pthresh = self.params.get("pthresh", 0.44)

        # Active covariance update for unsucessful candidates
        self.ancestors = list()

        # Constraint vectors for covariance adaptation
        # We work in the unit box [0, 1)
        self.constraints = numpy.zeros((self.dim * 2, self.dim))

        self.S_int = numpy.zeros(self.dim)
        for i, s in enumerate(self.space.steps()):
            if s is not None:
                self.S_int[i] = s

        self.i_I_R = numpy.flatnonzero(2 * self.sigma * numpy.diag(self.C) ** 0.5 < self.S_int)
        self.update_count = 0

    def _configure(self):
        self.d = self.params.get("d", 1.0 + self.dim / (2.0 * self.lambda_))
        self.ptarg = self.params.get("ptarg", 1.0 / (5 + numpy.sqrt(self.lambda_) / 2.0))
        self.cp = self.params.get("cp", self.ptarg * self.lambda_ / (2 + self.ptarg * self.lambda_))

        if self.update_count == 0:
            self.psucc = self.ptarg

    def _load_ancestors(self, results):
        # Get a list of the actual ancestor and the complementary information
        # on that ancestor
        ancestors = list()
        ancestors_ids = set()
        for c in sorted(self.conn.all_complementary(), key=itemgetter("_chocolate_id")):
            candidate = dict()
            candidate["step"] = numpy.array([c[str(k)] for k in self.space.names()])
            candidate["chocolate_id"] = c["_chocolate_id"]
            candidate["ancestor_id"] = c["_ancestor_id"]
            candidate["invalid"] = c["_invalid"]
            candidate["loss"] = None

            if c["_invalid"] == 0:
                candidate["X"] = numpy.array([results[c["_chocolate_id"]][str(k)] for k in self.space.names()])
                candidate["loss"] = results[c["_chocolate_id"]]["_loss"]

            ancestors.append(candidate)
            ancestors_ids.add(candidate["chocolate_id"])

        return ancestors, ancestors_ids

    def _load_bootstrap(self, results, ancestors_ids):
        # Find individuals produced by another algorithm
        bootstrap = list()
        for _, c in sorted(results.items()):
            # Skip those included in ancestors
            if c["_chocolate_id"] in ancestors_ids:
                continue

            candidate = dict()
            # The initial distribution is assumed uniform and centred on 0.5^n
            candidate["step"] = numpy.array([c[str(k)] - 0.5 for k in self.space.names()])
            candidate["X"] = numpy.array([results[c["_chocolate_id"]][str(k)] for k in self.space.names()])
            candidate["chocolate_id"] = c["_chocolate_id"]
            candidate["ancestor_id"] = -1
            # Compute constraint violation
            candidate["invalid"] = sum(2 ** (2 * i) for i, xi in enumerate(candidate["X"]) if xi < 0)
            candidate["invalid"] += sum(2 ** (2 * i + 1) for i, xi in enumerate(candidate["X"]) if xi >= 1)
            candidate["loss"] = None

            if candidate["invalid"] == 0:
                candidate["loss"] = c["_loss"]

            bootstrap.append(candidate)

        return bootstrap

    def _bootstrap(self, candidates):
        # Active covariance update for invalid individuals
        self._process_invalids(candidates)

        # Remove invalids and not evaluated
        candidates = [c for c in candidates if c["invalid"] == 0 and c["loss"] is not None]

        # Rank-mu update for covariance matrix
        if len(candidates) >= 4:
            mu = int(len(candidates) / 2)
            # superlinear weights (the usual default)
            weights = numpy.log(mu + 0.5) - numpy.log(numpy.arange(1, mu + 1))
            weights /= sum(weights)
            c1 = 2 / len(candidates[0]) ** 2
            cmu = mu / len(candidates[0]) ** 2

            candidates.sort(key=itemgetter("loss"))
            c_array = numpy.array([c["step"] for c in candidates[:mu]])
            cw = numpy.sum(weights * c_array.T, axis=1)

            self.pc = (1 - self.cc) * self.pc + numpy.sqrt(1 - (1 - self.cc) ** 2) * numpy.sqrt(mu) * cw
            self.C = (1 - c1 - cmu) * self.C + c1 * numpy.outer(self.pc, self.pc) + cmu * numpy.dot(weights * c_array.T,
                                                                                                    c_array)

            self.parent = candidates[0]

    def _update_internals(self, candidates):
        assert self.parent is not None, "No parent for CMA-ES internal update"
        assert "loss" in self.parent, "Parent has no loss in CMA-ES internal update"
        assert self.parent["loss"] is not None, "Invalid loss for CMA-ES parent"

        # Active covariance update for invalid individuals
        self._process_invalids(candidates)

        # Remove invalids and not evaluated
        candidates = [s for s in candidates if s["invalid"] == 0 and s["loss"] is not None]

        if len(candidates) == 0:
            # Empty group, abort
            return

        # Is the new point better than the parent?
        candidates.sort(key=itemgetter("loss"))
        lambda_succ = sum(s["loss"] <= self.parent["loss"] for s in candidates)
        p_succ = float(lambda_succ) / self.lambda_
        self.psucc = (1 - self.cp) * self.psucc + self.cp * p_succ

        # On success update the matrices C, A == B*D and evolution path
        if candidates[0]["loss"] <= self.parent["loss"]:
            self.parent = candidates[0].copy()
            if self.psucc < self.pthresh:
                self.pc = (1 - self.cc) * self.pc + numpy.sqrt(self.cc * (2 - self.cc)) * candidates[0]["step"]
                self.C = (1 - self.ccovp) * self.C + self.ccovp * numpy.outer(self.pc, self.pc)
            else:
                self.pc = (1 - self.cc) * self.pc
                self.C = (1 - self.ccovp) * self.C + self.ccovp * (numpy.outer(self.pc, self.pc)
                                                                   + self.cc * (2 - self.cc) * self.C)

            self.A = numpy.linalg.cholesky(self.C)

        elif len(self.ancestors) >= 5 and candidates[0]["loss"] > sorted(s["loss"] for s in self.ancestors)[-1]:
            # Active negative covariance update
            z = numpy.dot(numpy.linalg.inv(self.A), candidates[0]["step"])
            n_z2 = numpy.linalg.norm(z) ** 2
            if 1 - self.ccovn * n_z2 / (1 + self.ccovn) < 0.5:
                ccovn = 1 / (2 * numpy.linalg.norm(z) ** 2 - 1)
            else:
                ccovn = self.ccovn
            self.A = numpy.sqrt(1 + ccovn) * self.A + numpy.sqrt(1 + ccovn) / n_z2 * (
            numpy.sqrt(1 - ccovn * n_z2 / (1 + ccovn)) - 1) * numpy.dot(self.A, numpy.outer(z, z))
            self.C = numpy.dot(self.A, self.A.T)  # Yup we still have an update o C

        # Keep a list of ancestors sorted by order of appearance
        self.ancestors.insert(0, candidates[0])
        if len(self.ancestors) > 5:
            self.ancestors.pop(-1)

        # Update the step size
        self.sigma = self.sigma * numpy.exp(1.0 / self.d * (self.psucc - self.ptarg) / (1 - self.ptarg))

        # Update the dimensions where integer mutation is needed
        self.i_I_R = numpy.flatnonzero(2 * self.sigma * numpy.diag(self.C) ** 0.5 < self.S_int)
        self.update_count += 1

    def _process_invalids(self, candidates):
        # Process all invalid individuals
        for s in candidates:
            if s["invalid"] > 0:
                sum_vw = 0
                invalid_count = 0
                inv_A = numpy.linalg.inv(self.A)

                _, invalids = bin(s["invalid"]).split("b")

                for j, b in enumerate(reversed(invalids)):
                    if b == "1":
                        self.constraints[j, :] = (1 - self.cc) * self.constraints[j, :] + self.cc * s["step"]
                        w = numpy.dot(inv_A, self.constraints[j, :])
                        sum_vw += numpy.outer(self.constraints[j, :], w) / numpy.inner(w, w)
                        invalid_count += 1

                # Update A and make changes in C since in next updates we use C
                self.A = self.A - (self.beta / invalid_count) * sum_vw
                self.C = numpy.dot(self.A, self.A.T)

    def _generate(self):
        n_I_R = self.i_I_R.shape[0]
        R_int = numpy.zeros(self.dim)

        # Mixed integer CMA-ES is developped for (mu/mu , lambda)
        # We have a (1 + 1) setting, the integer will be probabilistic.
        # The integer mutation is lambda / 2 if all dimensions are integers or
        # min(lambda / 2 - 1, lambda / 10 + n_I_R + 1), minus 1 accounts for 
        # the last new candidate getting its integer mutation from the last best
        # solution. 
        if n_I_R == self.dim:
            p = 0.5
        else:
            p = min(0.5, 0.1 + n_I_R / self.dim)

        if n_I_R > 0 and numpy.random.rand() < p:
            Rp = numpy.zeros(self.dim)
            Rpp = numpy.zeros(self.dim)

            # Ri' has exactly one of its components set to one.
            # The Ri' are dependent in that the number of mutations for each coordinate
            # differs at most by one.
            j = numpy.random.choice(self.i_I_R)
            Rp[j] = 1
            Rpp[j] = numpy.random.geometric(p=0.7 ** (1.0 / n_I_R)) - 1

            I_pm1 = (-1) ** numpy.random.randint(0, 2, self.dim)
            R_int = I_pm1 * (Rp + Rpp)

        y = numpy.dot(numpy.random.standard_normal(self.dim), self.A.T)
        arz = self.parent["X"] + self.sigma * y + self.S_int * R_int

        return arz, y
