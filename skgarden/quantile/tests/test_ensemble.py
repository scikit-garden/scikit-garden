import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from skgarden.quantile import RandomForestQuantileRegressor
from skgarden.quantile import ExtraTreesQuantileRegressor
from skgarden.quantile import DecisionTreeQuantileRegressor
from skgarden.quantile import ExtraTreeQuantileRegressor

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, test_size=0.4, random_state=0)
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
estimators = [
    RandomForestQuantileRegressor(random_state=0),
    ExtraTreesQuantileRegressor(random_state=0)]


def test_quantile_attributes():
    for est in estimators:
        est.fit(X_train, y_train)

        # If a sample is not present in a particular tree, that
        # corresponding leaf is marked as -1.
        assert_array_equal(
            np.vstack(np.where(est.y_train_leaves_ == -1)),
            np.vstack(np.where(est.y_weights_ == 0))
        )

        # Should sum up to number of leaf nodes.
        assert_array_equal(
            np.sum(est.y_weights_, axis=1),
            [sum(tree.tree_.children_left == -1) for tree in est.estimators_]
        )

        n_est = est.n_estimators
        est.set_params(bootstrap=False)
        est.fit(X_train, y_train)
        assert_array_equal(
            np.sum(est.y_weights_, axis=1),
            [sum(tree.tree_.children_left == -1) for tree in est.estimators_]
        )
        assert np.all(est.y_train_leaves_ != -1)


def test_tree_forest_equivalence():
    """
    Test that a DecisionTree and RandomForest give equal quantile
    predictions when bootstrap is set to False.
    """
    rfqr = RandomForestQuantileRegressor(
        random_state=0, bootstrap=False, max_depth=2)
    rfqr.fit(X_train, y_train)

    dtqr = DecisionTreeQuantileRegressor(random_state=0, max_depth=2)
    dtqr.fit(X_train, y_train)

    assert np.all(rfqr.y_train_leaves_ == dtqr.y_train_leaves_)
    assert_array_almost_equal(
        rfqr.predict(X_test, quantile=10),
        dtqr.predict(X_test, quantile=10), 5)


def test_max_depth_None_rfqr():
    # Since each leaf is pure and has just one unique value.
    # the mean equals any quantile.
    rng = np.random.RandomState(0)
    X = rng.randn(10, 1)
    y = np.linspace(0.0, 100.0, 10.0)

    rfqr = RandomForestQuantileRegressor(
        random_state=0, bootstrap=False, max_depth=None)
    rfqr.fit(X, y)

    for quantile in [20, 40, 50, 60, 80, 90]:
        assert_array_almost_equal(
            rfqr.predict(X, quantile=None),
            rfqr.predict(X, quantile=quantile), 5)


def test_base_forest_quantile():
    """
    Test that the base estimators belong to the correct class.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(10, 1)
    y = np.linspace(0.0, 100.0, 10.0)

    rfqr = RandomForestQuantileRegressor(random_state=0, max_depth=1)
    rfqr.fit(X, y)
    for est in rfqr.estimators_:
        assert isinstance(est, DecisionTreeQuantileRegressor)

    etqr = ExtraTreesQuantileRegressor(random_state=0, max_depth=1)
    etqr.fit(X, y)
    for est in etqr.estimators_:
        assert isinstance(est, ExtraTreeQuantileRegressor)


def test_forest_toy_data():
    rng = np.random.RandomState(1)
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
