import numpy as np
import pandas as pd
from sklearn.tree import BaseDecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_X_y, check_array
from skgarden.quantile.utils import _quantile_tree_predict


class BaseTreeQuantileRegressor(BaseDecisionTree):
    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            The target values.

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = (n_samples, n_features), optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
            Returns self.
        """
        # y passed from a forest is 2-D. This is to silence the
        # annoying data-conversion warnings.
        y = np.asarray(y)
        if np.ndim(y) == 2 and y.shape[1] == 1:
            y = np.ravel(y)

        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=True)
        super(BaseTreeQuantileRegressor, self).fit(
            X, y, sample_weight=sample_weight, check_input=check_input,
            X_idx_sorted=X_idx_sorted)

        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        else:
            sample_weight = sample_weight.astype(np.float32)

        self.n_samples_ = len(y)
        self.y_train_ = y.reshape((-1, self.n_outputs_)).astype(np.float32)
        self.y_weights_ = sample_weight
        self.y_train_leaves_ = self.apply(X)

        return self

    def predict(self, X, q=None, check_input=True):
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        q : float, optional
            Value ranging from 0 to 1. By default, the mean is returned.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = (n_samples, n_outputs)
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if q is None:
            return super(BaseTreeQuantileRegressor, self).predict(X, check_input=check_input)
        elif q < 0 or q > 1:
            raise ValueError("Quantile should be provided in range 0 to 1")

        X_leaves = self.apply(X)
        return _quantile_tree_predict(X_leaves, self.y_train_, self.y_train_leaves_, self.y_weights_, q).squeeze()


class DecisionTreeQuantileRegressor(DecisionTreeRegressor, BaseTreeQuantileRegressor):
    """A decision tree regressor that provides quantile estimates.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    y_train_ : array-like
        Train target values.

    y_train_leaves_ : array-like.
        Cache the leaf nodes that each training sample falls into.
        y_train_leaves_[i] is the leaf that y_train[i] ends up at.
    """
    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None):
        super(DecisionTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)


class ExtraTreeQuantileRegressor(ExtraTreeRegressor, BaseTreeQuantileRegressor):
    def __init__(self,
                 criterion='mse',
                 splitter='random',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 random_state=None,
                 max_leaf_nodes=None):
        super(ExtraTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)
