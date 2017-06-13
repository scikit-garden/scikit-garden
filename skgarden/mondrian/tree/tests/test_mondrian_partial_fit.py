import numpy as np
from sklearn.utils.testing import assert_almost_equal
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
