What Algorithm to Choose?
=========================

The choice of the sampling/search strategy depends strongly on the problem tackled.
Ultimately, their are 4 aspects of the problem to look at:

  * the time required to evaluate a model,
  * the number of variables,
  * the type of variable (continuous or discrete),
  * the conditionality of the search space.

Chocolate proposes 5 algorithms with their own advantages and disadvantages:

  * :class:`~chocolate.Grid` sampling applies when all variables are discrete and the number
    of possibilities is low. A grid search will perform the exhaustive combinatorial search
    over all possibilities making the search extremely long even for medium sized problems.

  * :class:`~chocolate.Random` sampling is an alternative to grid search when the number of
    discrete parameters to optimize and the time required for each evaluation is high. When
    all parameters are discrete, random search will perform sampling without replacement making
    it an algorithm of choice when combinatorial exploration is not possible. With continuous
    parameters, it is preferable to use quasi random sampling.

  * :class:`~chocolate.QuasiRandom` sampling ensures a much more uniform exploration of the
    search space than traditional pseudo random. Thus, quasi random sampling is preferable
    when not all variables are discrete, the number of dimensions is high and the time
    required to evaluate a solution is high.

  * :class:`~chocolate.GaussianProcess` search models the search space using gaussian process
    regression, which allows to have an estimate of the loss function and the uncertainty on
    that estimate at every point of the search space. Modeling the search space suffers from
    the curse of dimensionality, which makes this method more suitable when the number of
    dimensions is low. Moreover, since it models both the expected loss and uncertainty, this
    search algorithm converges in few steps on superior configurations, making it a good choice
    when the time to complete the evaluation of a parameter configuration is high.

  * :class:`~chocolate.CMAES` search is one of the most powerful black-box optimization
    algorithm. However, it requires a significant number of model evaluation (in the order of
    10 to 50 times the number of dimensions) to converge to an optimal solution. This
    search method is more suitable when the time required for a model evaluation is relatively
    low.

In addition to the 5 previous algorithms Chocolate proposes a wrapper that transforms the
conditional search space problem in a `multi-armed bandit problem
<https://en.wikipedia.org/wiki/Multi-armed_bandit>`_.

  * :class:`~chocolate.ThompsonSampling` is a wrapper around any of the sampling/search
    algorithms that will allocate more resources to the exploration of the most promising
    subspaces. This method will help any of the algorithm in finding a superior solution
    in conditional search spaces.

Here is a table that resumes when to use each algorithm.

+-----------------------------------------+----------------+-------------------+---------------+----------------+
| Algorithm                               | Required time  | Dimensionality    | Continuity    | Conditionality |
+=========================================+================+===================+===============+================+
| :class:`~chocolate.Grid`                | Low            | Low               | All discrete  | Yes            |
+-----------------------------------------+----------------+-------------------+---------------+----------------+
| :class:`~chocolate.Random`              | Medium/High    | Medium/High       | All discrete  | Yes            |
+-----------------------------------------+----------------+-------------------+---------------+----------------+
| :class:`~chocolate.QuasiRandom`         | Medium/High    | Medium/High       | Mixed         | Yes            |
+-----------------------------------------+----------------+-------------------+---------------+----------------+
| :class:`~chocolate.GaussianProcess`     | Medium/High    | Low/Medium        | Mixed         | More or less   |
+-----------------------------------------+----------------+-------------------+---------------+----------------+
| :class:`~chocolate.CMAES`               | Low/Medium     | Low/Medium        | Mixed         | No             |
+-----------------------------------------+----------------+-------------------+---------------+----------------+
| :class:`~chocolate.ThompsonSampling`    | --             | --                | --            | Yes            |
+-----------------------------------------+----------------+-------------------+---------------+----------------+

