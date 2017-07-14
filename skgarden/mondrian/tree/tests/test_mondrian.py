"""
At fit time, the mondrian splitter works independent of labels.
So a lot of things can be factored between the MondrianTreeRegressor and
MondrianTreeClassifier
"""
import pickle
import numpy as np
from sklearn.base import clone
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_true

from skgarden.mondrian import MondrianTreeClassifier
from skgarden.mondrian import MondrianTreeRegressor

estimators = [MondrianTreeRegressor(random_state=0),
              MondrianTreeClassifier(random_state=0)]


def test_tree_predict():
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    y = [-1, -1, -1, 1, 1, 1]
    T = [[-1, -1], [2, 2], [3, 2]]

    # This test is dependent on the random-state since the feature
    # and the threshold selected at every split is independent of the
    # label.
    for est_true in estimators:
        est = clone(est_true)
        est.set_params(random_state=0, max_depth=1)
        est.fit(X, y)

        # mtr_tree = est.tree_
        cand_feature = est.tree_.feature[0]
        cand_thresh = est.tree_.threshold[0]
        assert_almost_equal(cand_thresh, -0.38669141)
        assert_almost_equal(cand_feature, 0.0)

        # Close to (1.0 / np.sum(np.max(X, axis=0) - np.min(X, axis=0)))
        assert_almost_equal(est.tree_.tau[0], 0.07112633)

        # For [-1, -1]:
        # P_not_separated = 1.0
        # Root:
        # eta_root = 0.0 (inside the bounding boc of the root)
        # P_root = 1 - exp(0.0) = 0.0
        # weight_root = P_root
        # mean_root = 0.0
        # Leaf:
        # P_not_separated = 1.0 * (1 - 0.0) = 1.0
        # weight_leaf = P_not_separated = 1.0
        # mean_leaf = -1.0

        # For regresssion:
        # prediction = weight_leaf * P_leaf = -1.0

        # For classifier:
        # proba = weight_leaf * P_leaf = [1.0, 0.0]

        # variance = (weight_root * (var_root + mean_root**2) +
        #             weight_leaf * (var_leaf + mean_leaf**2)) - mean**2
        # This reduces to weight_leaf * mean_leaf**2 - mean**2 = 1.0 * (1.0 - 1.0)
        # = 0.0

        # Similarly for [2, 2]:

        # For regression = weight_leaf * P_leaf = 1.0
        # prediction = 0.0 + 1.0
        # Variance reduces to zero

        # For classification
        # proba = weight_leaf * P_leaf = [0.0, 1.0]

        # For [3, 2]
        # P_not_separated = 1.0
        # Root:
        # Delta_root = 0.07112633
        # eta_root = 1.0
        # weight_root = 1 - exp(-0.07112633) = 0.0686
        # Leaf:
        # weight_leaf = P_not_separated = (1 - 0.0686) = 0.93134421

        # For regression:
        # prediction = mean_root * weight_root + mean_leaf * weight_leaf
        # prediction = 0.0 * 0.0686 + 0.93134421 * 1.0 = 0.93134421
        # For classification
        # proba = weight_root * P_root + weight_leaf * P_leaf
        # proba = 0.0686 * [0.5, 0.5] + 0.93134421 * [0.0 * 1.0]

        # variance = (weight_root * (var_root + mean_root**2) +
        #             weight_leaf * (var_leaf + mean_leaf**2)) - mean**2
        # = 0.0686 * (1 + 0) + 0.93134 * (0 + 1) - 0.93134421**2 = 0.132597

        if isinstance(est, RegressorMixin):
            T_predict, T_std = est.predict(T, return_std=True)
            assert_array_almost_equal(T_predict, [-1.0, 1.0, 0.93134421])
            assert_array_almost_equal(T_std, np.sqrt([0.0, 0.0, 0.132597]))
        else:
            last = (
                0.0686 * np.array([0.5, 0.5]) +
                0.93134421 * np.array([0.0 , 1.0])
            )
            T_proba = est.predict_proba(T)
            assert_array_almost_equal(
                T_proba,
                [[1.0, 0.0], [0.0, 1.0], last], 4)


def test_reg_boston():
    """Consistency on boston house prices"""
    mtr = MondrianTreeRegressor(random_state=0)
    boston = load_boston()
    X, y = boston.data, boston.target
    mtr.fit(X, y)
    score = mean_squared_error(mtr.predict(X), y)
    assert_less(score, 1, "Failed with score = {0}".format(score))

    mtr.partial_fit(X, y)
    score = mean_squared_error(mtr.predict(X), y)
    assert_less(score, 1, "Failed with score = {0}".format(score))


def test_array_repr():
    X = np.arange(10)[:, np.newaxis]
    y = np.arange(10)

    for est in estimators:
        new_est = clone(est)
        new_est.fit(X, y)
        new_est.partial_fit(X, y)


def test_pure_set():
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [1, 1, 1, 1, 1, 1]
    for est in estimators:
        est.fit(X, y)
        assert_array_almost_equal(est.predict(X), y)

        new_est = clone(est)
        new_est.partial_fit(X, y)
        assert_array_almost_equal(new_est.predict(X), y)


def test_numerical_stability():
    X = np.array([
        [152.08097839, 140.40744019, 129.75102234, 159.90493774],
        [142.50700378, 135.81935120, 117.82884979, 162.75781250],
        [127.28772736, 140.40744019, 129.75102234, 159.90493774],
        [132.37025452, 143.71923828, 138.35694885, 157.84558105],
        [103.10237122, 143.71928406, 138.35696411, 157.84559631],
        [127.71276855, 143.71923828, 138.35694885, 157.84558105],
        [120.91514587, 140.40744019, 129.75102234, 159.90493774]])

    y = np.array(
        [1., 0.70209277, 0.53896582, 0., 0.90914464, 0.48026916, 0.49622521])

    with np.errstate(all="raise"):
        for est in estimators:
            new_est = clone(est)
            if isinstance(est, ClassifierMixin):
                y_curr = np.round(y)
            else:
                y_curr = y
            new_est.fit(X, y_curr)
            new_est.fit(X, -y_curr)
            new_est.fit(-X, y_curr)
            new_est.fit(-X, -y_curr)
            new_est.partial_fit(X, y_curr)
            new_est.partial_fit(-X, y_curr)


def test_min_samples_split():
    iris = load_iris()
    X, y = iris.data, iris.target

    for est in estimators:
        est.set_params(min_samples_split=10, max_depth=None)
        est.fit(X, y)
        n_node_samples = est.tree_.n_node_samples[est.tree_.children_left != -1]
        assert_less(9, np.min(n_node_samples))


def test_tau():
    """
    Test time of split for the root.
    """
    X, y = make_regression(random_state=0, n_features=10)
    y = np.round(y)
    rate = np.sum(np.max(X, axis=0) - np.min(X, axis=0))

    for est in estimators:
        est = est.set_params(max_depth=1)
        taus = []
        for random_state in np.arange(100):
            est.set_params(random_state=random_state).fit(X, y)
            taus.append(est.tree_.tau[0])
        assert_almost_equal(np.mean(taus), 1.0 / rate, 2)


def test_dimension_location():
    """
    Test dimension and location of split.
    """
    rng = np.random.RandomState(0)
    X = rng.rand(100, 3)
    X[:, 1] *= 100
    X[:, 2] *= 50
    y = np.round(rng.randn(100))

    for est in estimators:
        n = 1000
        features = []
        thresholds = []
        for random_state in np.arange(1000):
            est.set_params(random_state=random_state).fit(X, y)
            features.append(est.tree_.feature[0])
            thresholds.append(est.tree_.threshold[0])

        # Check that this converges to the actual probability p of the bernoulli.
        diff = np.max(X, axis=0) - np.min(X, axis=0)
        p_act = diff / np.sum(diff)
        features = np.array(features)
        thresholds = np.array(thresholds)
        counts = np.bincount(features)
        p_sim = counts / np.sum(counts)
        assert_array_almost_equal(p_act, p_sim, 2)

        # Check that the split location converges to the (u + l) / 2 where
        # u and l are the upper and lower bounds of the feature.
        u = np.max(X, axis=0)[1]
        l = np.min(X, axis=0)[1]
        thresh_sim = np.mean(thresholds[features == 1])
        thresh_act = (u + l) / 2.0
        assert_array_almost_equal(thresh_act, thresh_sim, 1)


def load_scaled_boston():
    boston = load_boston()
    X, y = boston.data, boston.target
    n_train = 400
    n_test = 400
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[-n_test:], y[-n_test:]
    minmax = MinMaxScaler()
    X_train = minmax.fit_transform(X_train)
    X_test = minmax.transform(X_test)
    return X_train, X_test, y_train, y_test


def check_weighted_decision_path_train(est, X):
    leaf_nodes = est.apply(X)
    weights_sparse = est.weighted_decision_path(X)
    assert_array_equal(weights_sparse.data, np.ones(X.shape[0]))
    assert_array_equal(weights_sparse.indices, leaf_nodes)
    assert_array_equal(weights_sparse.indptr, np.arange(X.shape[0] + 1))


def test_weighted_decision_path_train():
    """
    Test the implementation of weighted_decision_path when all test points
    are in train points.
    """
    # Test that when all samples are in the training data all weights
    # should be concentrated at the leaf.
    X_train, _, y_train, _ = load_scaled_boston()
    y_train = np.round(y_train)
    for est in estimators:
        clone_est = clone(est)
        clone_est.fit(X_train, np.round(y_train))
        check_weighted_decision_path_train(clone_est, X_train)

        clone_est.partial_fit(X_train, np.round(y_train))
        check_weighted_decision_path_train(clone_est, X_train)


def check_weighted_decision_path_regression(mtr, X_test):
    weights = mtr.weighted_decision_path(X_test)
    node_means = mtr.tree_.mean
    node_variances = mtr.tree_.variance
    variances1 = []
    means1 = []

    for startptr, endptr in zip(weights.indptr[:-1], weights.indptr[1:]):
        curr_nodes = weights.indices[startptr: endptr]
        curr_weights = weights.data[startptr: endptr]
        curr_means = node_means[curr_nodes]
        curr_var = node_variances[curr_nodes]

        means1.append(np.sum(curr_weights * curr_means))
        variances1.append(np.sum(curr_weights * (curr_var + curr_means**2)))

    means1 = np.array(means1)
    variances1 = np.array(variances1)
    variances1 -= means1**2
    means2, std2 = mtr.predict(X_test, return_std=True)
    assert_array_almost_equal(means1, means2, 5)
    assert_array_almost_equal(variances1, std2**2, 3)


def test_weighted_decision_path_regression():
    X_train, X_test, y_train, y_test = load_scaled_boston()
    mtr = MondrianTreeRegressor(random_state=0)
    mtr.fit(X_train, y_train)
    check_weighted_decision_path_regression(mtr, X_test)
    mtr.partial_fit(X_train, y_train)
    check_weighted_decision_path_regression(mtr, X_test)


def check_weighted_decision_path_classif(mtc, X_test):
    weights = mtc.weighted_decision_path(X_test)
    node_probas = (
        mtc.tree_.value[:, 0, :] / np.expand_dims(mtc.tree_.n_node_samples, axis=1)
    )
    probas1 = []

    for startptr, endptr in zip(weights.indptr[:-1], weights.indptr[1:]):
        curr_nodes = weights.indices[startptr: endptr]
        curr_weights = np.expand_dims(weights.data[startptr: endptr], axis=1)
        curr_probas = node_probas[curr_nodes]
        probas1.append(np.sum(curr_weights * curr_probas, axis=0))

    probas2 = mtc.predict_proba(X_test)
    assert_array_almost_equal(probas1, probas2, 5)


def test_weighted_decision_path_classif():
    X_train, X_test, y_train, y_test = load_scaled_boston()
    y_train = np.round(y_train)
    y_test = np.round(y_test)

    mtc = MondrianTreeClassifier(random_state=0)
    mtc.fit(X_train, np.round(y_train))
    check_weighted_decision_path_classif(mtc, X_test)

    mtc.partial_fit(X_train, np.round(y_train))
    check_weighted_decision_path_classif(mtc, X_test)


def test_std_positive():
    """Sometimes variance can be slightly negative due to numerical errors."""
    X = np.linspace(-np.pi, np.pi, 20.0)
    y = 2*np.sin(X)
    X_train = np.reshape(X, (-1, 1))
    mr = MondrianTreeRegressor(random_state=0)
    mr.fit(X_train, y)

    X_test = np.array(
        [[2.87878788],
         [2.97979798],
         [3.08080808]])
    _, y_std = mr.predict(X_test, return_std=True)
    assert_false(np.any(np.isnan(y_std)))
    assert_false(np.any(np.isinf(y_std)))


def check_mean_std_reg_convergence(est, X_train, y_train):
    # For points completely in the training data and when
    # tree is grown to full depth.
    # mean should converge to the actual target value.
    # variance should converge to 0.0
    mean, std = est.predict(X_train, return_std=True)
    assert_array_almost_equal(mean, y_train, 5)
    assert_array_almost_equal(std, 0.0, 2)

    # For points completely far away from the training data, this
    # should converge to the empirical mean and variance.
    # X is scaled between to -1.0 and 1.0
    X_inf = np.vstack((20.0 * np.ones(X_train.shape[1]),
                       -20.0 * np.ones(X_train.shape[1])))
    inf_mean, inf_std = est.predict(X_inf, return_std=True)
    assert_array_almost_equal(inf_mean, y_train.mean(), 1)
    assert_array_almost_equal(inf_std, y_train.std(), 2)


def test_mean_std_reg_convergence():
    X_train, _, y_train, _ = load_scaled_boston()
    mr = MondrianTreeRegressor(random_state=0)
    mr.fit(X_train, y_train)
    check_mean_std_reg_convergence(mr, X_train, y_train)

    n_s = int(len(X_train) / 2)
    mr.partial_fit(X_train[:n_s], y_train[:n_s])
    mr.partial_fit(X_train[n_s:], y_train[n_s:])
    check_mean_std_reg_convergence(mr, X_train, y_train)


def check_proba_classif_convergence(X_train, y_train, mc):
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_train)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    proba = mc.predict_proba(X_train)
    labels = mc.predict(X_train)
    assert_array_equal(proba, y_bin)
    assert_array_equal(labels, lb.inverse_transform(y_bin))

    # For points completely far away from the training data, this
    # should converge to the empirical distribution of labels.
    # X is scaled between to -1.0 and 1.0
    X_inf = np.vstack((30.0 * np.ones(X_train.shape[1]),
                       -30.0 * np.ones(X_train.shape[1])))
    inf_proba = mc.predict_proba(X_inf)
    emp_proba = np.bincount(y_enc) / float(len(y_enc))
    assert_array_almost_equal(inf_proba, [emp_proba, emp_proba])


def test_proba_classif_convergence():
    X_train, _, y_train, _ = load_scaled_boston()
    y_train = np.round(y_train)
    mc = MondrianTreeClassifier(random_state=0)
    mc.fit(X_train, y_train)
    check_proba_classif_convergence(X_train, y_train, mc)

    mc.partial_fit(X_train, y_train)
    check_proba_classif_convergence(X_train, y_train, mc)


def check_tree_attributes(X, y, node_id, tree, check_impurity=True):
    """
    Recursive function to test the mean and variance at every node.
    """
    assert_almost_equal(np.var(y), tree.variance[node_id], 6)
    assert_almost_equal(np.mean(y), tree.mean[node_id], 6)
    if check_impurity:
        assert_almost_equal(np.var(y), tree.impurity[node_id])
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != -1:
        left_ind = X[:, tree.feature[node_id]] < tree.threshold[node_id]
        check_tree_attributes(
            X[left_ind], y[left_ind], left_child, tree, check_impurity)

    if right_child != -1:
        right_ind = X[:, tree.feature[node_id]] > tree.threshold[node_id]
        check_tree_attributes(
            X[right_ind], y[right_ind], right_child, tree, check_impurity)


def test_tree_attributes():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 5)
    y = np.sum(X[:, :4], axis=1)
    mr = MondrianTreeRegressor(random_state=0)
    mr.fit(X, y)
    check_tree_attributes(X, y, 0, mr.tree_)
    mr.partial_fit(X, y)
    check_tree_attributes(X, y, mr.tree_.root, mr.tree_, False)


def test_apply():
    X_train, X_test, y_train, y_test = load_scaled_boston()
    y_train = np.round(y_train)
    for est in estimators:
        est_clone = clone(est)
        est_clone.fit(X_train, y_train)
        train_leaves = est_clone.tree_.children_left[est_clone.apply(X_train)]
        test_leaves = est_clone.tree_.children_left[est_clone.apply(X_test)]
        assert_true(np.all(train_leaves == -1))
        assert_true(np.all(test_leaves == -1))

        est_clone.partial_fit(X_train, y_train)
        train_leaves = est_clone.tree_.children_left[est_clone.apply(X_train)]
        test_leaves = est_clone.tree_.children_left[est_clone.apply(X_test)]
        assert_true(np.all(train_leaves == -1))
        assert_true(np.all(test_leaves == -1))

def check_pickle(est, X, y):
    score1 = est.score(X, y)
    pickle_obj = pickle.dumps(est)

    est2 = pickle.loads(pickle_obj)
    assert_equal(type(est2), est.__class__)
    score2 = est2.score(X, y)
    assert_equal(score1, score2)


def test_pickle():
    X, _, y, _ = load_scaled_boston()
    y = np.round(y)

    for est in estimators:
        est.fit(X, y)
        check_pickle(est, X, y)
        est.partial_fit(X, y)
        check_pickle(est, X, y)


def test_tree_identical_labels():
    rng = np.random.RandomState(0)
    for est in estimators:
        X = rng.randn(100, 5)
        y = np.ones(100)
        c_est = clone(est)
        c_est.set_params(min_samples_split=2, max_depth=None)
        c_est.fit(X, y)
        assert_equal(c_est.tree_.n_node_samples, [100])
        if isinstance(c_est, ClassifierMixin):
            assert_equal(c_est.tree_.value, [[[100]]])
        else:
            assert_equal(c_est.tree_.value, [[[1.0]]])

        X = np.reshape(np.linspace(0.0, 1.0, 100), (-1, 1))
        y = np.array([0.0]*50 + [1.0]*50)
        c_est.fit(X, y)
        leaf_ids = c_est.tree_.children_left == -1
        assert_true(np.any(c_est.tree_.n_node_samples[leaf_ids] > 2))
