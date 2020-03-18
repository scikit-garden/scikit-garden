import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from numpy.testing import assert_array_almost_equal

from skgarden.quantile import DecisionTreeQuantileRegressor
from skgarden.quantile import ExtraTreeQuantileRegressor
from skgarden.quantile.utils import weighted_percentile

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, test_size=0.4, random_state=0)
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
estimators = [
    DecisionTreeQuantileRegressor(random_state=0),
    ExtraTreeQuantileRegressor(random_state=0)
]


def test_quantiles():
    # Test with max depth 1.
    for est in estimators:
        est.set_params(max_depth=1)
        est.fit(X_train, y_train)
        tree = est.tree_

        for q in [20, 40, 50, 60, 80, 90]:
            left_ind = X_train[:, tree.feature[0]] <= tree.threshold[0]
            right_ind = X_train[:, tree.feature[0]] > tree.threshold[0]

            # fixme
            left_q = weighted_percentile(y_train[left_ind], q)
            right_q = weighted_percentile(y_train[right_ind], q)

            for curr_X, curr_y in [[X_train, y_train], [X_test, y_test]]:
                actual_q = np.zeros(curr_X.shape[0])
                left_ind = curr_X[:, tree.feature[0]] <= tree.threshold[0]
                actual_q[left_ind] = left_q
                right_ind = curr_X[:, tree.feature[0]] > tree.threshold[0]
                actual_q[right_ind] = right_q

                expected_q = est.predict(curr_X, quantile=q)
                assert_array_almost_equal(expected_q, actual_q)


def test_max_depth_None():
    # Since each leaf is pure and has just one unique value.
    # the mean equals any quantile.
    for est in estimators:
        est.set_params(max_depth=None)
        est.fit(X_train, y_train)

        for quantile in [20, 40, 50, 60, 80, 90]:

            for curr_X in [X_train, X_test]:
                assert_array_almost_equal(
                    est.predict(curr_X, quantile=None),
                    est.predict(curr_X, quantile=quantile), 1)


def test_tree_toy_data():
    rng = np.random.RandomState(0)
    x1 = rng.randn(1, 10)
    X1 = np.tile(x1, (10000, 1))
    x2 = 20.0 * rng.randn(1, 10)
    X2 = np.tile(x2, (10000, 1))
    X = np.vstack((X1, X2))

    y1 = rng.randn(10000)
    y2 = 5.0 + rng.randn(10000)
    y = np.concatenate((y1, y2))

    for est in estimators:
        est.set_params(max_depth=1)
        est.fit(X, y)
        for quantile in [20, 30, 40, 50, 60, 70, 80]:
            assert_array_almost_equal(
                est.predict(x1, quantile=quantile),
                [np.percentile(y1, quantile)], 3)
            assert_array_almost_equal(
                est.predict(x2, quantile=quantile),
                [np.percentile(y2, quantile)], 3)
