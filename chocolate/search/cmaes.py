from itertools import groupby
from operator import itemgetter

import numpy

from ..base import SearchAlgorithm


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
        crossvalidation: A cross-validation object that handles experiment
            repetition.
        clear_db: If set to :data:`True` and a conflict arise between the
            provided space and the space in the database, completely clear the
            database and set the space to the provided one.
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

    def __init__(self, connection, space, crossvalidation=None, clear_db=False, **params):
        super(CMAES, self).__init__(connection, space, crossvalidation, clear_db)
        self.random_state = numpy.random.RandomState()
        self.params = params

    def _next(self, token=None):
        """Retrieve the next point to evaluate based on available data in the
        database. Each time :meth:`next` is called, the algorithm will reinitialize
        it-self based on the data in the database.

        Returns:
            A tuple containing a unique token and a fully qualified parameter set.
        """
        self._init()

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
        token = token or {}
        token.update({"_chocolate_id": self.conn.count_results()})

        # If the parent is still None, no information available
        if self.parent is None:
            # out = numpy.ones(self.dim) / 2.0
            out = self.random_state.rand(self.dim)

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
        # We have a (1 + 1) setting, thus we make the integer mutation probabilistic.
        # The integer mutation is lambda / 2 if all dimensions are integers or
        # min(lambda / 2 - 1, lambda / 10 + n_I_R + 1), minus 1 accounts for
        # the last new candidate getting its integer mutation from the last best
        # solution.
        if n_I_R == self.dim:
            p = 0.5
        else:
            p = min(0.5, 0.1 + n_I_R / self.dim)

        if n_I_R > 0 and self.random_state.rand() < p:
            Rp = numpy.zeros(self.dim)
            Rpp = numpy.zeros(self.dim)

            # Ri' has exactly one of its components set to one.
            # The Ri' are dependent in that the number of mutations for each coordinate
            # differs at most by one.
            j = self.random_state.choice(self.i_I_R)
            Rp[j] = 1
            Rpp[j] = self.random_state.geometric(p=0.7 ** (1.0 / n_I_R)) - 1

            I_pm1 = (-1) ** self.random_state.randint(0, 2, self.dim)
            R_int = I_pm1 * (Rp + Rpp)

        y = numpy.dot(self.random_state.standard_normal(self.dim), self.A.T)
        arz = self.parent["X"] + self.sigma * y + self.S_int * R_int

        return arz, y


class MOCMAES(SearchAlgorithm):
    """Multi-Objective Covariance Matrix Adaptation Evolution Strategy.

    """
    def __init__(self, connection, space, crossvalidation=None, clear_db=False, **params):
        super(MOCMAES, self).__init__(connection, space, crossvalidation, clear_db)
        self.random_state = numpy.random.RandomState()
        self.params = params

    def _next(self, token=None):
        pass

    def _generate(self):
        n_I_R = self.i_I_R.shape[0]
        R_int = numpy.zeros(self.dim)

        # Mixed integer CMA-ES is developped for (mu/mu , lambda)
        # We have a (1 + 1) setting, thus we make the integer mutation probabilistic.
        # The integer mutation is lambda / 2 if all dimensions are integers or
        # min(lambda / 2 - 1, lambda / 10 + n_I_R + 1), minus 1 accounts for
        # the last new candidate getting its integer mutation from the last best
        # solution.
        if n_I_R == self.dim:
            p = 0.5
        else:
            p = min(0.5, 0.1 + n_I_R / self.dim)

        if n_I_R > 0 and self.random_state.rand() < p:
            Rp = numpy.zeros(self.dim)
            Rpp = numpy.zeros(self.dim)

            # Ri' has exactly one of its components set to one.
            # The Ri' are dependent in that the number of mutations for each coordinate
            # differs at most by one.
            j = self.random_state.choice(self.i_I_R)
            Rp[j] = 1
            Rpp[j] = self.random_state.geometric(p=0.7 ** (1.0 / n_I_R)) - 1

            I_pm1 = (-1) ** self.random_state.randint(0, 2, self.dim)
            R_int = I_pm1 * (Rp + Rpp)

        y = numpy.dot(self.random_state.standard_normal(self.dim), self.A.T)

        # Select the parent at random from the non dominated set of parents
        ndom = sortLogNondominated(self.parents, len(self.parents), first_front_only=True)
        pi = numpy.random.randint(0, len(ndom))

        arz = self.parents[pi]["X"] + self.sigma[pi] * y + self.S_int * R_int

        return arz, y


####################
# Helper functions #
####################

def identity(obj):
    """Returns directly the argument *obj*.
    """
    return obj

def isDominated(wvalues1, wvalues2):
    """Returns whether or not *wvalues1* dominates *wvalues2*.

    :param wvalues1: The weighted fitness values that would be dominated.
    :param wvalues2: The weighted fitness values of the dominant.
    :returns: :obj:`True` if wvalues2 dominates wvalues1, :obj:`False`
              otherwise.
    """
    not_equal = False
    for self_wvalue, other_wvalue in zip(wvalues1, wvalues2):
        if self_wvalue > other_wvalue:
            return False
        elif self_wvalue < other_wvalue:
            not_equal = True
    return not_equal

def median(seq, key=identity):
    """Returns the median of *seq* - the numeric value separating the higher
    half of a sample from the lower half. If there is an even number of
    elements in *seq*, it returns the mean of the two middle values.
    """
    sseq = sorted(seq, key=key)
    length = len(seq)
    if length % 2 == 1:
        return key(sseq[(length - 1) // 2])
    else:
        return (key(sseq[(length - 1) // 2]) + key(sseq[length // 2])) / 2.0

def sortLogNondominated(individuals, k, first_front_only=False):
    """Sort *individuals* in pareto non-dominated fronts using the Generalized
    Reduced Run-Time Complexity Non-Dominated Sorting Algorithm presented by
    Fortin et al. (2013).

    :param individuals: A list of individuals to select from.
    :returns: A list of Pareto fronts (lists), with the first list being the
              true Pareto front.
    """
    if k == 0:
        return []

    #Separate individuals according to unique fitnesses
    unique_fits = defaultdict(list)
    for i, ind in enumerate(individuals):
        unique_fits[ind.fitness.wvalues].append(ind)

    #Launch the sorting algorithm
    obj = len(individuals[0].fitness.wvalues)-1
    fitnesses = unique_fits.keys()
    front = dict.fromkeys(fitnesses, 0)

    # Sort the fitnesses lexicographically.
    fitnesses.sort(reverse=True)
    sortNDHelperA(fitnesses, obj, front)

    #Extract individuals from front list here
    nbfronts = max(front.values())+1
    pareto_fronts = [[] for i in range(nbfronts)]
    for fit in fitnesses:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])

    # Keep only the fronts required to have k individuals.
    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= k:
                return pareto_fronts[:i+1]
        return pareto_fronts
    else:
        return pareto_fronts[0]

def sortNDHelperA(fitnesses, obj, front):
    """Create a non-dominated sorting of S on the first M objectives"""
    if len(fitnesses) < 2:
        return
    elif len(fitnesses) == 2:
        # Only two individuals, compare them and adjust front number
        s1, s2 = fitnesses[0], fitnesses[1]
        if isDominated(s2[:obj+1], s1[:obj+1]):
            front[s2] = max(front[s2], front[s1] + 1)
    elif obj == 1:
        sweepA(fitnesses, front)
    elif len(frozenset(map(itemgetter(obj), fitnesses))) == 1:
        #All individuals for objective M are equal: go to objective M-1
        sortNDHelperA(fitnesses, obj-1, front)
    else:
        # More than two individuals, split list and then apply recursion
        best, worst = splitA(fitnesses, obj)
        sortNDHelperA(best, obj, front)
        sortNDHelperB(best, worst, obj-1, front)
        sortNDHelperA(worst, obj, front)

def splitA(fitnesses, obj):
    """Partition the set of fitnesses in two according to the median of
    the objective index *obj*. The values equal to the median are put in
    the set containing the least elements.
    """
    median_ = median(fitnesses, itemgetter(obj))
    best_a, worst_a = [], []
    best_b, worst_b = [], []

    for fit in fitnesses:
        if fit[obj] > median_:
            best_a.append(fit)
            best_b.append(fit)
        elif fit[obj] < median_:
            worst_a.append(fit)
            worst_b.append(fit)
        else:
            best_a.append(fit)
            worst_b.append(fit)

    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))

    if balance_a <= balance_b:
        return best_a, worst_a
    else:
        return best_b, worst_b

def sweepA(fitnesses, front):
    """Update rank number associated to the fitnesses according
    to the first two objectives using a geometric sweep procedure.
    """
    stairs = [-fitnesses[0][1]]
    fstairs = [fitnesses[0]]
    for fit in fitnesses[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[fstair]+1)
        for i, fstair in enumerate(fstairs[idx:], idx):
            if front[fstair] == front[fit]:
                del stairs[i]
                del fstairs[i]
                break
        stairs.insert(idx, -fit[1])
        fstairs.insert(idx, fit)

def sortNDHelperB(best, worst, obj, front):
    """Assign front numbers to the solutions in H according to the solutions
    in L. The solutions in L are assumed to have correct front numbers and the
    solutions in H are not compared with each other, as this is supposed to
    happen after sortNDHelperB is called."""
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) == 0:
        #One of the lists is empty: nothing to do
        return
    elif len(best) == 1 or len(worst) == 1:
        #One of the lists has one individual: compare directly
        for hi in worst:
            for li in best:
                if isDominated(hi[:obj+1], li[:obj+1]) or hi[:obj+1] == li[:obj+1]:
                    front[hi] = max(front[hi], front[li] + 1)
    elif obj == 1:
        sweepB(best, worst, front)
    elif key(min(best, key=key)) >= key(max(worst, key=key)):
        #All individuals from L dominate H for objective M:
        #Also supports the case where every individuals in L and H
        #has the same value for the current objective
        #Skip to objective M-1
        sortNDHelperB(best, worst, obj-1, front)
    elif key(max(best, key=key)) >= key(min(worst, key=key)):
        best1, best2, worst1, worst2 = splitB(best, worst, obj)
        sortNDHelperB(best1, worst1, obj, front)
        sortNDHelperB(best1, worst2, obj-1, front)
        sortNDHelperB(best2, worst2, obj, front)

def splitB(best, worst, obj):
    """Split both best individual and worst sets of fitnesses according
    to the median of objective *obj* computed on the set containing the
    most elements. The values equal to the median are attributed so as
    to balance the four resulting sets as much as possible.
    """
    median_ = median(best if len(best) > len(worst) else worst, itemgetter(obj))
    best1_a, best2_a, best1_b, best2_b = [], [], [], []
    for fit in best:
        if fit[obj] > median_:
            best1_a.append(fit)
            best1_b.append(fit)
        elif fit[obj] < median_:
            best2_a.append(fit)
            best2_b.append(fit)
        else:
            best1_a.append(fit)
            best2_b.append(fit)

    worst1_a, worst2_a, worst1_b, worst2_b = [], [], [], []
    for fit in worst:
        if fit[obj] > median_:
            worst1_a.append(fit)
            worst1_b.append(fit)
        elif fit[obj] < median_:
            worst2_a.append(fit)
            worst2_b.append(fit)
        else:
            worst1_a.append(fit)
            worst2_b.append(fit)

    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))

    if balance_a <= balance_b:
        return best1_a, best2_a, worst1_a, worst2_a
    else:
        return best1_b, best2_b, worst1_b, worst2_b

def sweepB(best, worst, front):
    """Adjust the rank number of the worst fitnesses according to
    the best fitnesses on the first two objectives using a sweep
    procedure.
    """
    stairs, fstairs = [], []
    iter_best = iter(best)
    next_best = next(iter_best, False)
    for h in worst:
        while next_best and h[:2] <= next_best[:2]:
            insert = True
            for i, fstair in enumerate(fstairs):
                if front[fstair] == front[next_best]:
                    if fstair[1] > next_best[1]:
                        insert = False
                    else:
                        del stairs[i], fstairs[i]
                    break
            if insert:
                idx = bisect.bisect_right(stairs, -next_best[1])
                stairs.insert(idx, -next_best[1])
                fstairs.insert(idx, next_best)
            next_best = next(iter_best, False)

        idx = bisect.bisect_right(stairs, -h[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[h] = max(front[h], front[fstair]+1)
