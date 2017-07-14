# Most tests copied verbatim from sklearn.ensemble.tests.test_forest.py
import pickle
import numpy as np

from sklearn.base import clone
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_greater

from skgarden import MondrianForestClassifier
from skgarden import MondrianForestRegressor

boston = load_boston()
# The time of split and feature chosen for splitting are highly
# scale-sensitive.
scaler = MinMaxScaler()
X, y = boston.data, boston.target

y = np.round(y)
X = scaler.fit_transform(X)

ensembles = [
    MondrianForestRegressor(random_state=0),
    MondrianForestClassifier(random_state=0)]


def check_boston(est):
    score = est.score(X, y)
    assert_greater(score, 0.94, "Failed with score = %f" % score)


def test_boston():
    mr = MondrianForestRegressor(n_estimators=5, random_state=0)
    mr.fit(X, y)
    check_boston(mr)
    mr.partial_fit(X, y)
    check_boston(mr)


def test_forest_attributes():
    mr = MondrianForestRegressor(n_estimators=5, random_state=0)
    mr.fit([[1, 2, 3], [4, 5, 6]], [1, 2])
    assert_false(hasattr(mr, "classes_"))
    assert_false(hasattr(mr, "n_classes_"))

    mr.partial_fit([[1, 2, 3], [4, 5, 6]], [1, 2])
    assert_false(hasattr(mr, "classes_"))
    assert_false(hasattr(mr, "n_classes_"))

    mr = MondrianForestClassifier(n_estimators=5, random_state=0)
    mr.fit([[1, 2, 3], [4, 5, 6]], [1, 2])
    assert_true(hasattr(mr, "classes_"))
    assert_true(hasattr(mr, "n_classes_"))

    mr = MondrianForestClassifier(n_estimators=5, random_state=0)
    mr.partial_fit([[1, 2, 3], [4, 5, 6]], [1, 2])
    assert_true(hasattr(mr, "classes_"))
    assert_true(hasattr(mr, "n_classes_"))


def check_pickle(est):
    score1 = est.score(X, y)
    pickle_obj = pickle.dumps(est)

    est2 = pickle.loads(pickle_obj)
    assert_equal(type(est2), est.__class__)
    score2 = est2.score(X, y)
    assert_equal(score1, score2)


def test_pickle():
    for est1 in ensembles:
        est1.fit(X, y)
        check_pickle(est1)
        est1.partial_fit(X, y)
        check_pickle(est1)


def test_parallel_train():
    for curr_est in ensembles:
        est = clone(curr_est)
        y_pred = ([est.set_params(n_jobs=n_jobs).fit(X, y).predict(X)
                   for n_jobs in [1, 2, 4, 8]])
        for pred1, pred2 in zip(y_pred, y_pred[1:]):
            assert_array_equal(pred1, pred2)
        y_pred = ([est.set_params(n_jobs=n_jobs).partial_fit(X, y).predict(X)
                   for n_jobs in [1, 2, 4, 8]])
        for pred1, pred2 in zip(y_pred, y_pred[1:]):
            assert_array_equal(pred1, pred2)


def test_min_samples_split():
    min_samples_split = 4
    for curr_ensemble in ensembles:
        ensemble = clone(curr_ensemble)
        ensemble.set_params(
            min_samples_split=min_samples_split, n_estimators=100)
        ensemble.fit(X, y)
        for est in ensemble.estimators_:
            n_samples = est.tree_.n_node_samples
            leaves = est.tree_.children_left == -1
            assert_true(np.all(n_samples[~leaves] >= min_samples_split))


def test_memory_layout():
    for est in ensembles:
        for dtype in [np.float32, np.float64]:
            X_curr = np.asarray(X, dtype=dtype)
            assert_array_almost_equal(est.fit(X_curr, y).predict(X_curr), y, 3)
            assert_array_almost_equal(est.partial_fit(X_curr, y).predict(X_curr), y, 3)

            # C-order
            X_curr = np.asarray(X, order="C", dtype=dtype)
            assert_array_almost_equal(est.fit(X_curr, y).predict(X_curr), y, 3)
            assert_array_almost_equal(est.partial_fit(X_curr, y).predict(X_curr), y, 3)

            X_curr = np.asarray(X, order="F", dtype=dtype)
            assert_array_almost_equal(est.fit(X_curr, y).predict(X_curr), y, 3)
            assert_array_almost_equal(est.partial_fit(X_curr, y).predict(X_curr), y, 3)

            # Contiguous
            X_curr = np.ascontiguousarray(X_curr, dtype=dtype)
            assert_array_almost_equal(est.fit(X_curr, y).predict(X_curr), y, 3)
            assert_array_almost_equal(est.partial_fit(X_curr, y).predict(X_curr), y, 3)

            X_curr = np.array(X[::2], dtype=dtype)
            y_curr = np.asarray(y[::2])
            assert_array_almost_equal(
                est.fit(X_curr, y_curr).predict(X_curr), y_curr, 3)
            assert_array_almost_equal(
                est.partial_fit(X_curr, y_curr).predict(X_curr), y_curr, 3)


def check_decision_path(ensemble):
    indicator, col_inds = ensemble.decision_path(X)
    indices, indptr = indicator.indices, indicator.indptr

    n_nodes = [est.tree_.node_count for est in ensemble.estimators_]
    assert_equal(indicator.shape[0], X.shape[0])
    assert_equal(indicator.shape[1], sum(n_nodes))
    assert_array_equal(np.diff(col_inds), n_nodes)

    # Check that all leaf nodes are in the decision path.
    leaf_indices = ensemble.apply(X) + np.reshape(col_inds[:-1], (1, -1))
    for sample_ind, curr_leaf in enumerate(leaf_indices):
        sample_indices = indices[indptr[sample_ind]: indptr[sample_ind + 1]]
        assert_true(np.all(np.in1d(curr_leaf, sample_indices)))


def test_decision_path():
    for ensemble in ensembles:
        ensemble.fit(X, y)
        check_decision_path(ensemble)
        ensemble.partial_fit(X, y)
        check_decision_path(ensemble)


def check_weighted_decision_path(ensemble, X_train, X_test):
    # decision_path is implemented in sklearn while
    # weighted_decision_path is implemented here so check
    paths, col_inds = ensemble.decision_path(X_train)
    weight_paths, weight_col_inds = ensemble.weighted_decision_path(X_train)
    assert_array_equal(col_inds, weight_col_inds)

    n_nodes = [est.tree_.node_count for est in ensemble.estimators_]
    assert_equal(weight_paths.shape[0], X_train.shape[0])
    assert_equal(weight_paths.shape[1], sum(n_nodes))

    # We are calculating the weighted decision path on train data, so
    # the weights should be concentrated at the leaves.
    leaf_indices = ensemble.apply(X_train)
    for est_ind, curr_leaf_indices in enumerate(leaf_indices.T):
        curr_path = weight_paths[:, col_inds[est_ind]:col_inds[est_ind + 1]].toarray()
        assert_array_equal(np.where(curr_path)[1], curr_leaf_indices)

        # Sum of the weights across all the nodes in each estimator
        # for each sample should sum up to 1.0
        assert_array_almost_equal(
        np.ravel(ensemble.weighted_decision_path(X_test)[0].sum(axis=1)),
        ensemble.n_estimators * np.ones(X_test.shape[0]), 5)


def test_weighted_decision_path():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, test_size=0.4)

    for ensemble in ensembles:
        ensemble.fit(X_train, y_train)
        check_weighted_decision_path(ensemble, X_train, X_test)
        ensemble.partial_fit(X_train, y_train)
        check_weighted_decision_path(ensemble, X_train, X_test)


def check_mean_std_forest_regressor(est):
    # For points completely in the training data.
    # and max depth set to None.
    # mean should converge to the actual target value.
    # variance should converge to 0.0
    mean, std = est.predict(X, return_std=True)
    assert_array_almost_equal(mean, y, 5)
    assert_array_almost_equal(std, 0.0, 2)

    # For points completely far away from the training data, this
    # should converge to the empirical mean and variance.
    # X is scaled between to -1.0 and 1.0
    X_inf = np.vstack((30.0 * np.ones(X.shape[1]),
                       -30.0 * np.ones(X.shape[1])))
    inf_mean, inf_std = est.predict(X_inf, return_std=True)
    assert_array_almost_equal(inf_mean, y.mean(), 1)
    assert_array_almost_equal(inf_std, y.std(), 2)


def test_mean_std_forest_regressor():
    mfr = MondrianForestRegressor(random_state=0)
    mfr.fit(X, y)
    check_mean_std_forest_regressor(mfr)
    mfr.partial_fit(X, y)
    check_mean_std_forest_regressor(mfr)


def check_proba_classif_convergence(est, X_train, y_train):
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_train)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    proba = est.predict_proba(X_train)
    labels = est.predict(X_train)
    assert_array_equal(proba, y_bin)
    assert_array_equal(labels, lb.inverse_transform(y_bin))

    # For points completely far away from the training data, this
    # should converge to the empirical distribution of labels.
    X_inf = np.vstack((30.0 * np.ones(X_train.shape[1]),
                       -30.0 * np.ones(X_train.shape[1])))
    inf_proba = est.predict_proba(X_inf)
    emp_proba = np.bincount(y_enc) / float(len(y_enc))
    assert_array_almost_equal(inf_proba, [emp_proba, emp_proba], 3)


def test_proba_classif_convergence():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, test_size=0.4)
    mfc = MondrianForestClassifier(random_state=0)
    mfc.fit(X_train, y_train)
    check_proba_classif_convergence(mfc, X_train, y_train)
    mfc.partial_fit(X_train, y_train)
    check_proba_classif_convergence(mfc, X_train, y_train)


def test_tree_identical_labels():
    rng = np.random.RandomState(0)
    for ensemble in ensembles:
        X = rng.randn(100, 5)
        y = np.ones(100)
        ensemble.fit(X, y)
        for est in ensemble.estimators_:
            assert_equal(est.tree_.n_node_samples, [100])

            if isinstance(est, ClassifierMixin):
                assert_equal(est.tree_.value, [[[100]]])
            else:
                assert_equal(est.tree_.value, [[[1.0]]])

        X = np.reshape(np.linspace(0.0, 1.0, 100), (-1, 1))
        y = np.array([0.0]*50 + [1.0]*50)
        ensemble.fit(X, y)
        for est in ensemble.estimators_:
            leaf_ids = est.tree_.children_left == -1
            assert_true(np.any(est.tree_.n_node_samples[leaf_ids] > 2))
