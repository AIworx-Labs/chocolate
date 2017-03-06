from ..base import SearchAlgorithm
import numpy
from sklearn import gaussian_process
from scipy.optimize import minimize
from scipy.stats import norm
from bayes_opt import helpers

class GP(SearchAlgorithm):
    def __init__(self, connection, space, random_state=None, n_bootstrap=10, utility_function="ucb", kappa=2.756, xi=0.1, **params):
        super(GP, self).__init__(connection, space)
        self.params = params
        self.n_bootstrap = n_bootstrap
        if utility_function == "ucb":
            self.utility = self._ucb
            self.kappa = kappa
        elif utility_function == "ei":
            self.utility = self._ei
            self.xi = xi

        if isinstance(random_state, numpy.random.RandomState):
            self.random_state = random_state
        elif random_state is None:
            self.random_state = numpy.random
        else:
            self.random_state = numpy.random.RandomState(random_state)

    def next(self):
        with self.conn.lock():
            X, Xpending, y = self._load_database()
            token = {"_chocolate_id": len(X) + len(Xpending)}
            if len(X) < self.n_bootstrap:
                out = self.random_state.random_sample((len(list(self.space.names())),))
                # Signify the first point to others using loss set to None
                # Transform to dict with parameter names
                entry = {str(k) : v for k, v in zip(self.space.names(), out)}
                entry.update(token)
                self.conn.insert_result(entry)
                return token, self.space(out)

            gp, y = self._fit_gp(X, Xpending, y)
            out = self._acquisition(gp, y)
            entry = {str(k): v for k, v in zip(self.space.names(), out)}
            entry.update(token)
            self.conn.insert_result(entry)

            return token, self.space(out)

    def _fit_gp(self, X, Xpending, y):
        gp = gaussian_process.GaussianProcessRegressor()
        X = numpy.array([[elem[k] for k in self.space.names()] for elem in X])
        Xpending = numpy.array([[elem[k] for k in self.space.names()] for elem in Xpending])
        y = numpy.array(y)
        gp.fit(X, y)
        if Xpending.size:
            y_predict = gp.predict(Xpending)
            X = numpy.concatenate([X, Xpending])
            y = numpy.concatenate([y, y_predict])
            gp.fit(X, y)
        return gp, y

    def _load_database(self):
        dbdata = self.conn.all_results()
        X, y, Xpending = [], [], []
        for elem in dbdata:
            if elem["_loss"]:
                X.append(elem)
                y.append(-elem["_loss"])  # Negative because we maximize
            else:
                Xpending.append(elem)
        return X, Xpending, y

    def _acquisition(self, gp, y):
        y_max = y.max()
        bounds = numpy.array([[0., 0.999999] for _ in self.space.names()])
        max_acq = None#y_max
        x_max = None
        x_seeds = numpy.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(100, bounds.shape[0]))
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            x_try[[not i for i in self.space.isactive(x_try)]] = 0
            res = minimize(lambda x: -self.utility(x.reshape(1, -1), gp=gp, y_max=y_max, kappa=self.kappa),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")

            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]
        assert x_max is not None, "This is an invalid case"
        return x_max

    @staticmethod
    def _ucb(x, gp, kappa, **kwargs):
        mean, std = gp.predict(x, return_std=True)
        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi, **kwargs):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
