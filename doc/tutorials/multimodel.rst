Optimizing Over Multiple Models
===============================

Searching for a good configuration for multiple models at the same time is
possible using conditional search spaces. A conditional search space is 
defined by a list of dictionaries each containing one or more
non-:class:`chocolate.Distribution` parameter. 

Independent Parameter Search
----------------------------

Say we want to optimize the
hyperparameters of `SVMs <http://scikit-learn.org/stable/modules/svm.html>`_
with different kernels or even multiple types of SVMs. We would define the
:ref:`search space <Search Space Representation>`  as a list of dictionaries,
one for each model. ::

    from sklearn.svm import SVC, LinearSVC
    import chocolate as choco

    space = [{"algo" : SVC, "kernel" : "rbf",
                  "C" : choco.log(low=-2, high=10, base=10),
                  "gamma" : choco.log(low=-9, high=3, base=10)},
             {"algo" : SVC, "kernel" : "poly",
                  "C" : choco.log(low=-2, high=10, base=10),
                  "gamma" : choco.log(low=-9, high=3, base=10),
                  "degree" : choco.quantized_uniform(low=1, high=5, step=1),
                  "coef0" : choco.uniform(low=-1, high=1)},
             {"algo" : LinearSVC,
                  "C" : choco.log(low=-2, high=10, base=10),
                  "penalty" : choco.choice(["l1", "l2"])}]

Lets now define the optimization function. Since we were able to directly
define the classifier type as the parameter ``"algo"`` we can use that directly.
Note that the F1 score has to be maximized, however, Chocolate always minimizes
the loss. Thus, we shall return the negative of the F1 score.::

    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    def score_svm(X, y, algo, **params):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf = algo(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        return -f1_score(y_test, y_pred)

Just as in the simpler examples, we will load our dataset, make our
connection and explore the configurations using one of the algorithm ::

    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=80000, random_state=1)

    conn = choco.SQLiteConnection(url="sqlite:///db.db")
    sampler = choco.QuasiRandom(conn, space, random_state=42, skip=0)

    token, params = sampler.next()
    loss = score_svm(X, y, **params)
    sampler.update(token, loss)

And just like that multiple models are explored simultaneously.


Sharing Parameters Between Models
---------------------------------

In the last example, all SVMs share parameter ``C``. Some optimizers might
take advantage of this information to optimize this parameters across all SVM
types thus accelerating convergence to an optimal configuration. Furthermore,
the ``SVC`` algorithm and parameter ``gamma`` are shared for ``"rbf"`` and
``"poly"`` kernels. We can rewrite the last search space using nested
dictionaries to represent the multiple condition levels. ::

    space = {"algo" : {SVC : {"gamma" : choco.log(low=-9, high=3, base=10)},
                              "kernel" : {"rbf" : None,
                                          "poly" : {"degree" : choco.quantized_uniform(low=1, high=5, step=1),
                                                    "coef0" : choco.uniform(low=-1, high=1)}},
                       LinearSVC : {"penalty" : choco.choice(["l1", "l2"])}},
             "C" : choco.log(low=-2, high=10, base=10)}

We can still add complexity to the search space by combining multiple
dictionaries at the top level, if for example a configuration does not name an
``"algo"`` parameter. ::

    space = [{"algo" : {SVC : {"gamma" : choco.log(low=-9, high=3, base=10)},
                               "kernel" : {"rbf" : None,
                                          "poly" : {"degree" : choco.quantized_uniform(low=1, high=5,  step=1),
                                                     "coef0" : choco.uniform(low=-1, high=1)}},
                        LinearSVC : {"penalty" : choco.choice(["l1", "l2"])}},
              "C" : choco.log(low=-2, high=10, base=10)},

             {"type" : "an_other_optimizer", "param" : choco.uniform(low=-1, high=1)}]

The remaining of the exploration is identical to the previous section.
