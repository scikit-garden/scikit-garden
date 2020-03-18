import numpy as np
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y
from joblib import delayed, Parallel

from ..tree import MondrianTreeClassifier
from ..tree import MondrianTreeRegressor

from ...forest import ForestClassifier
from ...forest import ForestRegressor

def _single_tree_pfit(tree, X, y, classes=None):
    if classes is not None:
        tree.partial_fit(X, y, classes)
    else:
        tree.partial_fit(X, y)
    return tree

class BaseMondrian(object):
    def weighted_decision_path(self, X):
        """
        Returns the weighted decision path in the forest.

        Each non-zero value in the decision path determines the
        weight of that particular node while making predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input.

        Returns
        -------
        decision_path : sparse csr matrix, shape = (n_samples, n_total_nodes)
            Return a node indicator matrix where non zero elements
            indicate the weight of that particular node in making predictions.

        est_inds : array-like, shape = (n_estimators + 1,)
            weighted_decision_path[:, est_inds[i]: est_inds[i + 1]]
            provides the weighted_decision_path of estimator i
        """
        X = self._validate_X_predict(X)
        est_inds = np.cumsum(
            [0] + [est.tree_.node_count for est in self.estimators_])
        paths = sparse.hstack(
            [est.weighted_decision_path(X) for est in self.estimators_]).tocsr()
        return paths, est_inds

    # XXX: This is mainly a stripped version of BaseForest.fit
    # from sklearn.forest
    def partial_fit(self, X, y, classes=None):
        """
        Incremental building of Mondrian Forests.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        y: array_like, shape = [n_samples]
            Input targets.

        classes: array_like, shape = [n_classes]
            Ignored for a regression problem. For a classification
            problem, if not provided this is inferred from y.
            This is taken into account for only the first call to
            partial_fit and ignored for subsequent calls.

        Returns
        -------
        self: instance of MondrianForest
        """
        X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        random_state = check_random_state(self.random_state)

        # Wipe out estimators if partial_fit is called after fit.
        first_call = not hasattr(self, "first_")
        if first_call:
            self.first_ = True

        if isinstance(self, ClassifierMixin):
            if first_call:
                if classes is None:
                    classes = LabelEncoder().fit(y).classes_

                self.classes_ = classes
                self.n_classes_ = len(self.classes_)

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        self.n_outputs_ = 1

        # Initialize estimators at first call to partial_fit.
        if first_call:
            # Check estimators
            self._validate_estimator()
            self.estimators_ = []

            for _ in range(self.n_estimators):
                tree = self._make_estimator(append=False, random_state=random_state)
                self.estimators_.append(tree)

        # XXX: Switch to threading backend when GIL is released.
        if isinstance(self, ClassifierMixin):
            self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_single_tree_pfit)(t, X, y, classes) for t in self.estimators_)
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_single_tree_pfit)(t, X, y) for t in self.estimators_)

        return self


class MondrianForestRegressor(ForestRegressor, BaseMondrian):
    """
    A MondrianForestRegressor is an ensemble of MondrianTreeRegressors.

    The variance in predictions is reduced by averaging the predictions
    from all trees.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    max_depth : integer, optional (default=None)
        The depth to which each tree is grown. If None, the tree is either
        grown to full depth or is constrained by `min_samples_split`.

    min_samples_split : integer, optional (default=2)
        Stop growing the tree if all the nodes have lesser than
        `min_samples_split` number of samples.

    bootstrap : boolean, optional (default=False)
        If bootstrap is set to False, then all trees are trained on the
        entire training dataset. Else, each tree is fit on n_samples
        drawn with replacement from the training dataset.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self,
                 n_estimators=10,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(MondrianForestRegressor, self).__init__(
            base_estimator=MondrianTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("max_depth", "min_samples_split",
                              "random_state"),
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        """Builds a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        return super(MondrianForestRegressor, self).fit(X, y)

    def predict(self, X, return_std=False):
        """
        Returns the predicted mean and std.

        The prediction is a GMM drawn from
        \(\sum_{i=1}^T w_i N(m_i, \sigma_i)\) where \(w_i = {1 \over T}\).

        The mean \(E[Y | X]\) reduces to \({\sum_{i=1}^T m_i \over T}\)

        The variance \(Var[Y | X]\) is given by $$Var[Y | X] = E[Y^2 | X] - E[Y | X]^2$$
        $$=\\frac{\sum_{i=1}^T E[Y^2_i| X]}{T} - E[Y | X]^2$$
        $$= \\frac{\sum_{i=1}^T (Var[Y_i | X] + E[Y_i | X]^2)}{T} - E[Y| X]^2$$

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input samples.

        return_std : boolean, default (False)
            Whether or not to return the standard deviation.

        Returns
        -------
        y : array-like, shape = (n_samples,)
            Predictions at X.

        std : array-like, shape = (n_samples,)
            Standard deviation at X.
        """
        X = check_array(X)
        if not hasattr(self, "estimators_"):
            raise NotFittedError("The model has to be fit before prediction.")
        ensemble_mean = np.zeros(X.shape[0])
        exp_y_sq = np.zeros_like(ensemble_mean)

        for est in self.estimators_:
            if return_std:
                mean, std = est.predict(X, return_std=True)
                exp_y_sq += (std**2 + mean**2)
            else:
                mean = est.predict(X, return_std=False)
            ensemble_mean += mean

        ensemble_mean /= len(self.estimators_)
        exp_y_sq /= len(self.estimators_)

        if not return_std:
            return ensemble_mean
        std = exp_y_sq - ensemble_mean**2
        std[std <= 0.0] = 0.0
        std **= 0.5
        return ensemble_mean, std

    def partial_fit(self, X, y):
        """
        Incremental building of Mondrian Forest Regressors.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        y: array_like, shape = [n_samples]
            Input targets.

        classes: array_like, shape = [n_classes]
            Ignored for a regression problem. For a classification
            problem, if not provided this is inferred from y.
            This is taken into account for only the first call to
            partial_fit and ignored for subsequent calls.

        Returns
        -------
        self: instance of MondrianForestClassifier
        """
        return super(MondrianForestRegressor, self).partial_fit(X, y)


class MondrianForestClassifier(ForestClassifier, BaseMondrian):
    """
    A MondrianForestClassifier is an ensemble of MondrianTreeClassifiers.

    The probability \(p_{j}\) of class \(j\) is given
    $$\sum_{i}^{N_{est}} \\frac{p_{j}^i}{N_{est}}$$

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    max_depth : integer, optional (default=None)
        The depth to which each tree is grown. If None, the tree is either
        grown to full depth or is constrained by `min_samples_split`.

    min_samples_split : integer, optional (default=2)
        Stop growing the tree if all the nodes have lesser than
        `min_samples_split` number of samples.

    bootstrap : boolean, optional (default=False)
        If bootstrap is set to False, then all trees are trained on the
        entire training dataset. Else, each tree is fit on n_samples
        drawn with replacement from the training dataset.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self,
                 n_estimators=10,
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(MondrianForestClassifier, self).__init__(
            base_estimator=MondrianTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("max_depth", "min_samples_split",
                              "random_state"),
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        """Builds a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=np.float32, multi_output=False)
        return super(MondrianForestClassifier, self).fit(X, y)

    def partial_fit(self, X, y, classes=None):
        """
        Incremental building of Mondrian Forest Classifiers.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        y: array_like, shape = [n_samples]
            Input targets.

        classes: array_like, shape = [n_classes]
            Ignored for a regression problem. For a classification
            problem, if not provided this is inferred from y.
            This is taken into account for only the first call to
            partial_fit and ignored for subsequent calls.

        Returns
        -------
        self: instance of MondrianForestClassifier
        """
        return super(MondrianForestClassifier, self).partial_fit(
            X, y, classes=classes)
