# Most tests copied verbatim from sklearn.ensemble.tests.test_forest.py
import pickle
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_greater

from mondrian_forest import MondrianForestRegressor

boston = load_boston()
# The time of split and feature chosen for splitting are highly
# scale-sensitive.
scaler = MinMaxScaler()
X, y = boston.data, boston.target
X = scaler.fit_transform(X)

def test_boston():
    mr = MondrianForestRegressor(n_estimators=5, random_state=0)
    mr.fit(X, y)
    score = mr.score(X, y)
    assert_greater(score, 0.94, "Failed with score = %f" % score)


def test_regressor_attributes():
    mr = MondrianForestRegressor(n_estimators=5, random_state=0)
    assert_false(hasattr(mr, "classes"))
    assert_false(hasattr(mr, "n_classes_"))

    mr.fit([[1, 2, 3], [4, 5, 6]], [1, 2])
    assert_false(hasattr(mr, "classes"))
    assert_false(hasattr(mr, "n_classes_"))


def test_pickle():
    mr1 = MondrianForestRegressor(random_state=0)
    mr1.fit(X, y)
    score1 = mr1.score(X, y)
    pickle_obj = pickle.dumps(mr1)

    mr2 = pickle.loads(pickle_obj)
    assert_equal(type(mr2), mr1.__class__)
    score2 = mr2.score(X, y)
    assert_equal(score1, score2)


def test_parallel_train():
    mr = MondrianForestRegressor(n_estimators=20, random_state=0, max_depth=4)
    y_pred = (
        [mr.set_params(n_jobs=n_jobs).fit(X, y).predict(X) for n_jobs in [1, 2, 4, 8]]
    )
    for pred1, pred2 in zip(y_pred, y_pred[1:]):
        assert_array_equal(pred1, pred2)


def test_min_samples_split():
    min_samples_split = 5
    mr = MondrianForestRegressor(
        random_state=0, n_estimators=100, min_samples_split=min_samples_split)
    mr.fit(X, y)
    for est in mr.estimators_:
        n_samples = est.tree_.n_node_samples
        leaves = est.tree_.children_left == -1
        assert_true(np.all(n_samples[~leaves] >= min_samples_split))
        imp_leaves = np.logical_and(leaves, est.tree_.variance > 1e-7)
        assert_true(np.all(n_samples[imp_leaves] < min_samples_split))


def test_memory_layout():
    mr = MondrianForestRegressor(random_state=0)

    for dtype in [np.float32, np.float64]:
        X_curr = np.asarray(X, dtype=dtype)
        assert_array_almost_equal(mr.fit(X_curr, y).predict(X_curr), y, 3)

        # C-order
        X_curr = np.asarray(X, order="C", dtype=dtype)
        assert_array_almost_equal(mr.fit(X_curr, y).predict(X_curr), y, 3)

        X_curr = np.asarray(X, order="F", dtype=dtype)
        assert_array_almost_equal(mr.fit(X_curr, y).predict(X_curr), y, 3)

        # Contiguous
        X_curr = np.ascontiguousarray(X_curr, dtype=dtype)
        assert_array_almost_equal(mr.fit(X_curr, y).predict(X_curr), y, 3)

        X_curr = np.array(X[::2], dtype=dtype)
        y_curr = np.asarray(y[::2])
        assert_array_almost_equal(
            mr.fit(X_curr, y_curr).predict(X_curr), y_curr, 3)


def test_decision_path():
    mr = MondrianForestRegressor(random_state=0)
    mr.fit(X, y)
    indicator, col_inds = mr.decision_path(X)
    indices, indptr, data = indicator.indices, indicator.indptr, indicator.data

    n_nodes = [est.tree_.node_count for est in mr.estimators_]
    assert_equal(indicator.shape[0], X.shape[0])
    assert_equal(indicator.shape[1], sum(n_nodes))
    assert_array_equal(np.diff(col_inds), n_nodes)

    # Check that all leaf nodes are in the decision path.
    leaf_indices = mr.apply(X) + np.reshape(col_inds[:-1], (1, -1))
    for sample_ind, curr_leaf in enumerate(leaf_indices):
        sample_indices = indices[indptr[sample_ind]: indptr[sample_ind + 1]]
        assert_true(np.all(np.in1d(curr_leaf, sample_indices)))


def test_weighted_decision_path():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, test_size=0.4)
    mr = MondrianForestRegressor(random_state=0)
    mr.fit(X_train, y_train)

    # decision_path is implemented in sklearn while
    # weighted_decision_path is implemented here so check
    paths, col_inds = mr.decision_path(X_train)
    weight_paths, weight_col_inds = mr.weighted_decision_path(X_train)
    assert_array_equal(col_inds, weight_col_inds)

    n_nodes = [est.tree_.node_count for est in mr.estimators_]
    assert_equal(weight_paths.shape[0], X_train.shape[0])
    assert_equal(weight_paths.shape[1], sum(n_nodes))

    # We are calculating the weighted decision path on train data, so
    # the weights should be concentrated at the leaves.
    leaf_indices = mr.apply(X_train)
    for est_ind, curr_leaf_indices in enumerate(leaf_indices.T):
        curr_path = weight_paths[:, col_inds[est_ind]:col_inds[est_ind + 1]].toarray()
        assert_array_equal(np.where(curr_path)[1], curr_leaf_indices)

    # Sum of the weights across all the nodes in each estimator
    # for each sample should sum up to 1.0
    assert_array_almost_equal(
        np.ravel(mr.weighted_decision_path(X_test)[0].sum(axis=1)),
        mr.n_estimators * np.ones(X_test.shape[0]), 5)

def test_mean_std():
    mr = MondrianForestRegressor(random_state=0)
    mr.fit(X, y)

    # For points completely in the training data.
    # mean should converge to the actual target value.
    # variance should converge to 0.0
    mean, std = mr.predict(X, return_std=True)
    assert_array_almost_equal(mean, y, 5)
    assert_array_almost_equal(std, 0.0, 2)

    # For points completely far away from the training data, this
    # should converge to the empirical mean and variance.
    # X is scaled between to -1.0 and 1.0
    X_inf = np.vstack((20.0 * np.ones(X.shape[1]),
                       -20.0 * np.ones(X.shape[1])))
    inf_mean, inf_std = mr.predict(X_inf, return_std=True)
    assert_array_almost_equal(inf_mean, y.mean(), 1)
    assert_array_almost_equal(inf_std, y.std(), 2)
