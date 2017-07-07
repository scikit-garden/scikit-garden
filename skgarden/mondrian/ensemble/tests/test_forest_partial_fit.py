import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.datasets import load_digits
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal

from skgarden import MondrianForestRegressor
from skgarden import MondrianForestClassifier
from skgarden.mondrian.tree.tests.test_mondrian_partial_fit import check_partial_fit_max_depth


def check_partial_fit_equivalence(size_batch, f, random_state, X, y, is_clf=False):
    start_ptr = list(range(0, 100, size_batch))
    end_ptr = start_ptr[1:] + [100]
    if not is_clf:
        p_f = MondrianForestRegressor(random_state=random_state)
    else:
        p_f = MondrianForestClassifier(random_state=random_state)
    for start, end in zip(start_ptr, end_ptr):
        p_f.partial_fit(X[start:end], y[start:end])
    for est, p_est in zip(f.estimators_, p_f.estimators_):
        assert_array_equal(p_est.tree_.n_node_samples, est.tree_.n_node_samples)
        assert_array_equal(p_est.tree_.threshold, est.tree_.threshold)
        assert_array_equal(p_est.tree_.feature, est.tree_.feature)
        assert_equal(p_est.tree_.root, est.tree_.root)
        assert_array_equal(p_est.tree_.value, est.tree_.value)
        assert_equal(est.tree_.n_node_samples[est.tree_.root], 100)
        assert_equal(p_est.tree_.n_node_samples[est.tree_.root], 100)


def test_partial_fit_equivalence():
    X, y = make_regression(random_state=0, n_samples=100)
    mfr = MondrianForestRegressor(random_state=0)
    mfr.partial_fit(X, y)
    for batch_size in [10, 20, 25, 50, 90]:
        check_partial_fit_equivalence(batch_size, mfr, 0, X, y)

    X, y = make_classification(random_state=0, n_samples=100)
    mtc = MondrianForestClassifier(random_state=0)
    mtc.partial_fit(X, y)
    for batch_size in [10, 20, 25, 50, 90]:
        check_partial_fit_equivalence(batch_size, mtc, 0, X, y, is_clf=True)


def check_fit_after_partial_fit(ensemble, X, y):
    ensemble.fit(X, y)
    for est in ensemble.estimators_:
        assert_equal(est.tree_.n_node_samples[0], 10)

    ensemble.partial_fit(X, y)
    for est in ensemble.estimators_:
        assert_equal(est.tree_.n_node_samples[est.tree_.root], 10)

    ensemble.partial_fit(X, y)
    for est in ensemble.estimators_:
        assert_equal(est.tree_.n_node_samples[est.tree_.root], 20)


def test_fit_after_partial_fit():
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)
    y = np.floor(rng.randn(10))
    mfr = MondrianForestRegressor(random_state=0)
    check_fit_after_partial_fit(mfr, X, y)

    mfc = MondrianForestClassifier(random_state=0)
    check_fit_after_partial_fit(mfc, X, y)


def test_partial_fit_max_depth():
    X_c, y_c = load_digits(return_X_y=True)
    X_r, y_r = make_regression(n_samples=10000)
    for d in list(np.arange(1, 22)) + [None]:
        mfr = MondrianForestRegressor(random_state=0, max_depth=d)
        mfr.partial_fit(X_r[:5000], y_r[:5000])
        mfr.partial_fit(X_r[5000:], y_r[5000:])
        for est in mfr.estimators_:
            check_partial_fit_max_depth(est, d, X_r.shape[0])

        mfc = MondrianForestClassifier(random_state=0, max_depth=d)
        mfc.partial_fit(X_c[:X_c.shape[0] // 2], y_c[:y_c.shape[0] // 2])
        mfc.partial_fit(X_c[X_c.shape[0] // 2:], y_c[y_c.shape[0] // 2:])
        for est in mfc.estimators_:
            check_partial_fit_max_depth(est, d, X_c.shape[0])
