from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import chocolate as choco


def score_gbt(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Chocolate minimizes the loss
    return -r2_score(y_test, y_pred)


def main():
    X, y = load_boston(return_X_y=True)

    # Connect to sqlite database in current directory
    conn = choco.SQLiteConnection(url="sqlite:///gbt-boston.db")
    s = {"learning_rate": choco.uniform(0.001, 0.1),
         "n_estimators": choco.quantized_uniform(25, 525, 1),
         "max_depth": choco.quantized_uniform(2, 25, 1),
         "subsample": choco.uniform(0.7, 1.0)}

    sampler = choco.QuasiRandom(conn, s, seed=110, skip=3)
    for _ in range(50):
        token, params = sampler.next()
        loss = score_gbt(X, y, params)
        sampler.update(token, loss)

if __name__ == "__main__":
    main()
