Realistic Example
-----------------

Lets see how one can optimize the hyper parameters of say a `gradient boosting
tree classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ense
mble.GradientBoostingClassifier.html>`_ using scikit-learn and Chocolate.
First we'll do the necessary imports. ::

    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    import chocolate as choco

And we'll define our train function to return the negative of the
`F1 score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_
as loss function for our Chocolate minimizer. Nothing fancy here. ::

    def score_gbt(X, y, params):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        gbt = GradientBoostingClassifier(**params)
        gbt.fit(X_train, y_train)
        y_pred = gbt.predict(X_test)

        return -f1_score(y_test, y_pred)


Then we will load our dataset (or `make <http://scikit-learn.org/stable/module
s/generated/sklearn.datasets.make_classification.html>`_ it). ::

    X, y = make_classification(n_samples=80000, random_state=1)

And just as in the :ref:`Basics` tutorial, we'll decide where the data is
stored and the :ref:`search space <Search Space Representation>` for the
algorithm. We will optimize over a mix of continuous and discrete variables. ::

    conn = choco.SQLiteConnection(url="sqlite:///db.db")
    s = {"learning_rate" : choco.uniform(0.001, 0.1),
         "n_estimators"  : choco.quantized_uniform(25, 525, 25),
         "max_depth"     : choco.quantized_uniform(2, 10, 2),
         "subsample"     : choco.quantized_uniform(0.7, 1.05, 0.05)}

Finally, we will define our sampling algorithm, ::

    sampler = choco.QuasiRandom(conn, s, random_state=110, skip=3)

request a set of parameters to test, ::

    token, params = sampler.next()

get the loss for that set, ::

    loss = score_gbt(X, y, params)

and signify it to the database. ::

    sampler.update(token, loss)


Running this ~20 line script a bunch of times in completely separate processes
will explore the search space to find a good parameter set for your problem. As
simple as that.