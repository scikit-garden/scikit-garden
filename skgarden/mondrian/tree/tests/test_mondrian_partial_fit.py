"""
Tests specific to incremental building of trees.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.datasets import load_digits
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_greater

from skgarden import MondrianTreeClassifier
from skgarden import MondrianTreeRegressor


def check_partial_fit_one_sample(tree):
    assert_array_equal(tree.threshold, [-2])
    assert_array_equal(tree.feature, [-2])
    assert_array_equal(tree.children_left, [-1])
    assert_array_equal(tree.children_right, [-1])
    assert_array_equal(tree.tau, [np.inf])
    assert_array_equal(tree.n_node_samples, [1])


def test_partial_fit_one_sample():
    rng = np.random.RandomState(0)
    X = rng.randn(1, 5)
    y = [4.5]
    mtr = MondrianTreeRegressor(random_state=0)
    mtr.partial_fit(X, y)
    assert_array_equal(mtr.tree_.value, [[[4.5]]])
    assert_array_equal(mtr.tree_.variance, [0.0])
    check_partial_fit_one_sample(mtr.tree_)

    y = [1]
    mtc = MondrianTreeClassifier(random_state=0)
    mtc.partial_fit(X, y, classes=[0, 1])
    check_partial_fit_one_sample(mtr.tree_)


def check_partial_fit_two_samples(tree, X):
    assert_array_equal(tree.n_node_samples, [1, 1, 2])
    s_f = tree.feature[-1]
    s_t = tree.threshold[-1]
    sort_thresh = np.sort([X[0, s_f], X[1, s_f]])
    assert_greater(s_t, sort_thresh[0])
    assert_greater(sort_thresh[1], s_t)
    if X[0, s_f] < s_t:
        assert_array_equal(tree.children_left, [-1, -1, 0])
        assert_array_equal(tree.children_right, [-1, -1, 1])
    else:
        assert_array_equal(tree.children_left, [-1, -1, 1])
        assert_array_equal(tree.children_right, [-1, -1, 0])


def test_partial_fit_two_samples():
    rng = np.random.RandomState(10)
    X = rng.randn(2, 5)
    y = rng.randn(2)
    for r in range(10):
        mtr = MondrianTreeRegressor(random_state=r)
        mtr.partial_fit(X, y)
        tree = mtr.tree_
        assert_array_almost_equal(tree.value[:, 0, 0], [y[0], y[1], np.mean(y)])
        assert_array_almost_equal(tree.variance, [0, 0, np.var(y)])
        check_partial_fit_two_samples(tree, X)

    y = [0, 1]
    for r in range(10):
        mtc = MondrianTreeClassifier(random_state=r)
        mtc.partial_fit(X, y)
        tree = mtc.tree_
        assert_array_almost_equal(tree.value[:, 0, :], [[1, 0], [0, 1], [1, 1]])
        check_partial_fit_two_samples(tree, X)


def check_and_return_children(tree, node, val, var=None):
    if var is not None:
        assert_almost_equal(tree.variance[node], var)
    assert_array_almost_equal(tree.value[node], val)
    l_id = tree.children_left[node]
    r_id = tree.children_right[node]
    return l_id, r_id


def test_partial_fit_toy_data1():
    X = [[2.0, 1.0, 3.0],
         [-1.0, 2.0, 2.0],
         [1.0, 1.5, 2.5],   # inside the bounds of the first two samples.
         [10.0, 5.0, 6.0]]  # induces a split and creates a new root.

    #             [0, 1, 2, 3]
    #                 /\
    #                /  \
    #          [0, 1, 2] [3]
    #            (d=2, f=2.3608)
    #              /\
    #             / \
    #          [1]    [0, 2]
    #                 / \
    #              (d=0, f=1.17251138)
    #                [2]   [0]
    X = np.array(X)
    mtr = MondrianTreeRegressor(random_state=1)
    y_reg = [2, 1, 3, 4]
    mtr.partial_fit(X, y_reg)
    tree_reg = mtr.tree_

    y_clf = [0, 1, 2, 0]
    mtc = MondrianTreeClassifier(random_state=1)
    mtc.partial_fit(X, y_clf)
    tree_clf = mtc.tree_

    l, r = check_and_return_children(
        tree_reg, tree_reg.root, np.mean(y_reg), np.var(y_reg))
    ll, lr = check_and_return_children(
        tree_reg, l, np.mean(y_reg[:3]), np.var(y_reg[:3]))
    check_and_return_children(tree_reg, r, 4.0, 0.0)
    check_and_return_children(tree_reg, ll, 1.0, 0.0)
    lrl, lrr = check_and_return_children(
        tree_reg, lr, (y_reg[0] + y_reg[2]) / 2.0,
        np.var([y_reg[0], y_reg[2]]))
    check_and_return_children(tree_reg, lrl, y_reg[2], 0.0)
    check_and_return_children(tree_reg, lrr, y_reg[0], 0.0)

    l, r = check_and_return_children(tree_clf, tree_clf.root, [[2, 1, 1]])
    ll, lr = check_and_return_children(tree_clf, l, [[1, 1, 1]])
    check_and_return_children(tree_clf, r, [[1, 0, 0]])
    check_and_return_children(tree_clf, ll, [[0, 1, 0]])
    lrl, lrr = check_and_return_children(tree_clf, lr, [[1, 0, 1]])
    check_and_return_children(tree_clf, lrl, [[0, 0, 1]])
    check_and_return_children(tree_clf, lrr, [[1, 0, 0]])


def test_partial_fit_toy_data2():
    X = [[2.0, 1.0, 3.0],
         [-1.0, 2.0, 2.0],
         [11.0, 7.0, 4.5],
         [10.0, 5.0, 6.0]]
    X = np.array(X)

    #            [0, 1, 2, 3]
    #                /\
    #               /  \
    #          [0, 1]  [2, 3]
    #    (d=2, f=2.36) (d=1, f=5.345)
    #          /\        /\
    #         / \       / \
    #       [1] [0]    [3]  [2]

    y_reg = [2, 1, 3, 4]
    mtr = MondrianTreeRegressor(random_state=1)
    mtr.partial_fit(X, y_reg)
    tree = mtr.tree_
    l, r = check_and_return_children(
        tree, tree.root, np.mean(y_reg), np.var(y_reg))
    ll, lr = check_and_return_children(
        tree, l, np.mean(y_reg[:2]), np.var(y_reg[:2]))
    rl, rr = check_and_return_children(
        tree, r, np.mean(y_reg[2:]), np.var(y_reg[:2]))
    check_and_return_children(tree, ll, y_reg[1], 0.0)
    check_and_return_children(tree, lr, y_reg[0], 0.0)
    check_and_return_children(tree, rl, y_reg[3], 0.0)
    check_and_return_children(tree, rr, y_reg[2], 0.0)

    y_clf = [0, 1, 1, 2]
    mtc = MondrianTreeClassifier(random_state=1)
    mtc.partial_fit(X, y_clf)
    tree = mtc.tree_
    l, r = check_and_return_children(tree, tree.root, [[1, 2, 1]])
    ll, lr = check_and_return_children(tree, l, [[1, 1, 0]])
    rl, rr = check_and_return_children(tree, r, [[0, 1, 1]])
    check_and_return_children(tree, ll, [[0, 1, 0]])
    check_and_return_children(tree, lr, [[1, 0, 0]])
    check_and_return_children(tree, rl, [[0, 0, 1]])
    check_and_return_children(tree, rr, [[0, 1, 0]])


def test_mondrian_tree_n_node_samples():
    for r in range(1000):
        X, y = make_regression(n_samples=2, random_state=r)
        mtr = MondrianTreeRegressor(random_state=0)
        mtr.partial_fit(X, y)
        assert_array_equal(mtr.tree_.n_node_samples, [1, 1, 2])


def check_partial_fit_equivalence(size_batch, est, random_state, X, y, is_clf=False):
    start_ptr = list(range(0, 100, size_batch))
    end_ptr = start_ptr[1:] + [100]
    if not is_clf:
        p_est = MondrianTreeRegressor(random_state=random_state)
    else:
        p_est = MondrianTreeClassifier(random_state=random_state)
    for start, end in zip(start_ptr, end_ptr):
        p_est.partial_fit(X[start:end], y[start:end])
    assert_array_equal(p_est.tree_.n_node_samples, est.tree_.n_node_samples)
    assert_array_equal(p_est.tree_.threshold, est.tree_.threshold)
    assert_array_equal(p_est.tree_.feature, est.tree_.feature)
    assert_equal(p_est.tree_.root, est.tree_.root)
    assert_array_equal(p_est.tree_.value, est.tree_.value)


def test_partial_fit_equivalence():
    X, y = make_regression(random_state=0, n_samples=100)
    mtr = MondrianTreeRegressor(random_state=0)
    mtr.partial_fit(X, y)
    for batch_size in [10, 20, 25, 50, 90]:
        check_partial_fit_equivalence(batch_size, mtr, 0, X, y)

    X, y = make_classification(random_state=0, n_samples=100)
    mtc = MondrianTreeClassifier(random_state=0)
    mtc.partial_fit(X, y)
    for batch_size in [10, 20, 25, 50, 90]:
        check_partial_fit_equivalence(batch_size, mtc, 0, X, y, is_clf=True)


def check_partial_fit_duplicates(est, values):
    assert_array_equal(est.tree_.n_node_samples, [100])
    assert_almost_equal(est.tree_.value, values)
    assert_array_equal(est.tree_.children_left, [-1])
    assert_array_equal(est.tree_.children_right, [-1])
    assert_array_equal(est.tree_.root, 0)


def test_partial_fit_duplicates():
    rng = np.random.RandomState(0)
    X = rng.randn(1, 100)
    X_dup = np.tile(X, (100, 1))
    y = [2] * 100
    mtr = MondrianTreeRegressor(random_state=0)
    mtr.partial_fit(X_dup, y)
    check_partial_fit_duplicates(mtr, [[[2.0]]])

    mtc = MondrianTreeClassifier(random_state=0)
    mtc.partial_fit(X_dup, y, classes=[1, 2])
    check_partial_fit_duplicates(mtc, [[[0.0, 100.0]]])


def check_fit_after_partial_fit(est, X, y):
    est.fit(X, y)
    assert_equal(est.tree_.n_node_samples[0], 10)
    est.partial_fit(X, y)
    assert_equal(est.tree_.n_node_samples[est.tree_.root], 10)
    est.partial_fit(X, y)
    assert_equal(est.tree_.n_node_samples[est.tree_.root], 20)


def test_fit_after_partial_fit():
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)
    y = np.floor(rng.randn(10))
    mtr = MondrianTreeRegressor(random_state=0)
    check_fit_after_partial_fit(mtr, X, y)

    mtc = MondrianTreeClassifier(random_state=0)
    check_fit_after_partial_fit(mtc, X, y)


def check_online_fit(clf, X, y, batch_size, is_clf=True):
    start_ptr = np.arange(0, len(y), batch_size)
    end_ptr = list(start_ptr[1:]) + [len(y)]

    for start, end in zip(start_ptr, end_ptr):
        if is_clf:
            if start == 0:
                classes = np.unique(y)
            else:
                classes = None
            clf.partial_fit(X[start:end], y[start:end], classes=classes)
        else:
            clf.partial_fit(X[start:end], y[start:end])
    assert_almost_equal(clf.score(X, y), 1.0)


def test_partial_fit_n_samples_1000():
    mtc = MondrianTreeClassifier(random_state=0)
    X, y = load_digits(return_X_y=True)
    check_online_fit(mtc, X, y, 20)

    mtc = MondrianTreeClassifier(random_state=0)
    check_online_fit(mtc, X, y, 100)

    X, y = make_regression(random_state=0, n_samples=10000)
    mtr = MondrianTreeRegressor(random_state=0)
    check_online_fit(mtr, X, y, 100, is_clf=False)

    mtr = MondrianTreeRegressor(random_state=0)
    check_online_fit(mtr, X, y, 20, is_clf=False)


def test_min_samples_split():
    X_c, y_c = load_digits(return_X_y=True)
    X_r, y_r = make_regression(n_samples=10000, random_state=0)

    for mss in [2, 4, 10, 20]:
        mtr = MondrianTreeRegressor(random_state=0, min_samples_split=mss)
        mtr.partial_fit(X_r[: X_r.shape[0] // 2], y_r[: X_r.shape[0] // 2])
        mtr.partial_fit(X_r[X_r.shape[0] // 2:], y_r[X_r.shape[0] // 2:])
        n_node_samples = mtr.tree_.n_node_samples[mtr.tree_.children_left != -1]
        assert_greater(np.min(n_node_samples) + 1, mss)

        mtc = MondrianTreeClassifier(random_state=0, min_samples_split=mss)
        mtc.partial_fit(X_c[: X_c.shape[0] // 2], y_c[: X_c.shape[0] // 2])
        mtc.partial_fit(X_c[X_c.shape[0] // 2:], y_c[X_c.shape[0] // 2:])
        n_node_samples = mtc.tree_.n_node_samples[mtc.tree_.children_left != -1]
        assert_greater(np.min(n_node_samples) + 1, mss)
