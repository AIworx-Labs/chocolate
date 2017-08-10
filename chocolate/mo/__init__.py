from collections import defaultdict

import numpy

try:
    # try importing the C version
    from . import _hv as hv
except ImportError:
    # fallback on python version
    from . import _pyhv as hv


def argsortNondominated(losses, k, first_front_only=False):
    """Sort the first *k* *losses* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    losses.

    :param losses: A list of losses to select from.
    :param k: The number of elements to select.
    :param first_front_only: If :obj:`True` sort only the first front and
                             exit.
    :returns: A list of Pareto fronts (lists) containing the individuals
              index.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    loss2c = defaultdict(list)
    for i, c in enumerate(losses):
        loss2c[tuple(c)].append(i)
    losses_keys = list(loss2c.keys())

    current_front = []
    next_front = []
    dominating_losses = defaultdict(int)
    dominated_losses = defaultdict(list)

    # Rank first Pareto front
    for i, li in enumerate(losses_keys):
        for lj in losses_keys[i+1:]:
            if dominates(li, lj):
                dominating_losses[lj] += 1
                dominated_losses[li].append(lj)
            elif dominates(lj, li):
                dominating_losses[li] += 1
                dominated_losses[lj].append(li)
        if dominating_losses[li] == 0:
            current_front.append(li)

    fronts = [[]]
    for loss in current_front:
        fronts[0].extend(loss2c[loss])
    pareto_sorted = len(fronts[0])

    if first_front_only:
        return fronts[0]

    # Rank the next front until at least the requested number
    # candidates are sorted
    N = min(len(losses), k)
    while pareto_sorted < N:
        fronts.append([])
        for lp in current_front:
            for ld in dominated_losses[lp]:
                dominating_losses[ld] -= 1
                if dominating_losses[ld] == 0:
                    next_front.append(ld)
                    pareto_sorted += len(loss2c[ld])
                    fronts[-1].extend(loss2c[ld])
        current_front = next_front
        next_front = []

    return fronts


def dominates(loss1, loss2, obj=slice(None)):
    """Returns wether or not loss1 dominates loss2, while minimizing all
    objectives.
    """
    not_equal = False
    for l1i, l2i in zip(loss1[obj], loss2[obj]):
        if l1i < l2i:
            not_equal = True
        elif l1i > l2i:
            return False
    return not_equal


def hypervolume_indicator(front, **kargs):
    """Returns the index of the individual with the least hypervolume
    contribution. The provided *front* should be a set of non-dominated
    individuals having each a :attr:`fitness` attribute.
    """
    # Hypervolume use implicit minimization
    obj = numpy.array([c["loss"] for c in front])
    ref = kargs.get("ref", None)
    if ref is None:
        ref = numpy.max(obj, axis=0) + 1

    def contribution(i):
        # The contribution of point p_i in point set P
        # is the hypervolume of P without p_i
        return hv.hypervolume(numpy.concatenate((obj[:i], obj[i+1:])), ref)

    contrib_values = map(contribution, range(len(front)))

    # Select the maximum hypervolume value (correspond to the minimum difference)
    return numpy.argmax(contrib_values)