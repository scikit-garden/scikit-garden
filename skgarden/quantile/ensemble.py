from __future__ import division

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.utils import check_array, check_X_y

from ..forest import ForestRegressor
from .utils import _quantile_forest_predict, _weighted_random_sample, generate_sample_indices, set_tdigest, \
    tdigestlist_quantile
import dask.dataframe as dd
import dask.array as da


class _DefaultForestQuantileRegressor(ForestRegressor):
    def fit(self, X, y, sample_weight=None):
        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=True)
        super(_DefaultForestQuantileRegressor, self).fit(X, y, sample_weight=sample_weight)

        self.n_samples_ = len(y)
        self.y_train_ = y.reshape((-1, self.n_outputs_)).astype(np.float32)
        self.y_train_leaves_ = self.apply(X).T
        self.y_weights_ = np.zeros_like(self.y_train_leaves_, dtype=np.float32)

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        for i, est in enumerate(self.estimators_):
            if self.verbose > 0:
                print(f"Tree\t{i + 1}\tof\t{self.n_estimators}")

            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(
                    est.random_state, self.n_samples_)
            else:
                bootstrap_indices = np.arange(self.n_samples_)

            df = pd.DataFrame({"weights": np.bincount(bootstrap_indices, minlength=self.n_samples_) * sample_weight,
                               "leaves": self.y_train_leaves_[i, :]})

            self.y_weights_[i, :] = df.weights.values / df.groupby("leaves") \
                .transform("sum", engine="numba", engine_kwargs={"parallel": True}) \
                .values.squeeze()

        self.y_train_leaves_[self.y_weights_ == 0] = -1
        return self

    def predict(self, X, q=None):
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if q is None:
            return super(_DefaultForestQuantileRegressor, self).predict(X)
        elif q < 0 or q > 1:
            raise ValueError("Quantile should be provided in range 0 to 1")
        else:
            q = np.float32(q)

        X_leaves = self.apply(X)
        return _quantile_forest_predict(X_leaves, self.y_train_, self.y_train_leaves_, self.y_weights_, q).squeeze()


class _RandomSampleForestQuantileRegressor(ForestRegressor):
    def fit(self, X, y, sample_weight=None):
        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=True)
        super(_RandomSampleForestQuantileRegressor, self).fit(X, y, sample_weight=sample_weight)

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        self.n_samples_ = len(y)
        y = y.reshape((-1, self.n_outputs_))

        for i, est in enumerate(self.estimators_):
            if self.verbose:
                print(f"Tree\t{i + 1}\tof\t{self.n_estimators}")

            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(
                    est.random_state, self.n_samples_)
            else:
                bootstrap_indices = np.arange(self.n_samples_)

            weights = np.bincount(bootstrap_indices, minlength=self.n_samples_) * sample_weight
            mask = weights > 0

            leaves = est.apply(X[mask])
            idx = np.arange(len(leaves), dtype=np.int64)
            y_masked = y[mask]

            unique_leaves, sampled_idx = _weighted_random_sample(leaves, weights[mask], idx)

            est.training_data_ = pd.DataFrame(y_masked[sampled_idx], index=unique_leaves)

        return self

    def predict(self, X, q=None):
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if q is None:
            return super(_RandomSampleForestQuantileRegressor, self).predict(X)

        quantiles = np.empty((len(X), self.n_outputs_, self.n_estimators))
        for i, est in enumerate(self.estimators_):
            quantiles[:, :, i] = est.training_data_.loc[est.apply(X)].values

        return np.quantile(quantiles, q=q, axis=-1).squeeze()


class _TDigestForestQuantileRegressor(ForestRegressor):
    def fit(self, X, y, sample_weight=None):
        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=True)
        super(_TDigestForestQuantileRegressor, self).fit(X, y, sample_weight=sample_weight)

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        self.n_samples_ = len(y)
        self.features = list(range(self.n_outputs_))
        y = y.reshape((-1, self.n_outputs_))

        for i, est in enumerate(self.estimators_):
            if self.verbose > 0:
                print(f"Tree\t{i + 1}\tof\t{self.n_estimators}")

            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(
                    est.random_state, self.n_samples_)
            else:
                bootstrap_indices = np.arange(self.n_samples_)

            df = pd.DataFrame({"weights": np.bincount(bootstrap_indices, minlength=self.n_samples_) * sample_weight},
                              index=est.apply(X))
            df.loc[:, self.features] = y
            df.index.name = "leaves"

            df = df[df.weights > 0].groupby("leaves", sort=False, group_keys=False) \
                      .apply(lambda x: pd.Series(
                                         [set_tdigest(x[i].values, x.weights.values) for i in self.features]))

            est.training_data_ = df

        return self

    def predict(self, X, q=None):
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if q is None:
            return super(_TDigestForestQuantileRegressor, self).predict(X)

        quantile_stack = []

        for i, est in enumerate(self.estimators_):
            quantile_stack.append(est.training_data_.loc[est.apply(X)].reset_index())

        df_quantiles = pd.concat(quantile_stack, axis=1)
        quantiles = np.empty((len(X), self.n_outputs_))

        for i in range(self.n_outputs_):
            quantiles[:, i] = df_quantiles[i].apply(lambda x: tdigestlist_quantile(x, q), axis=1).values

        return quantiles.squeeze()


class _ForestQuantileRegressor:
    """
    Intermediate class, used for the mixing of the right regressor inheritance and
    base_estimator selection. It also updates the __repr__ function to represent
    both the 'method' and base_estimator.
    """
    # allowed options
    methods = ['default', 'sample', 'tdigest']
    base_estimators = ['random_forest', 'extra_trees']

    def __new__(cls, method='default', type='random_forest', **kwargs):
        if method == 'default':
            base = _DefaultForestQuantileRegressor
        elif method == 'sample':
            base = _RandomSampleForestQuantileRegressor
        elif method == 'tdigest':
            base = _TDigestForestQuantileRegressor
        else:
            raise ValueError(f"Method not recognised, should be one of {_ForestQuantileRegressor.methods}")

        if type == 'random_forest':
            base_estimator = DecisionTreeRegressor()
        elif type == 'extra_trees':
            base_estimator = ExtraTreeRegressor()
        else:
            raise ValueError(f"Type not recognised, should be one of {_ForestQuantileRegressor.base_estimators}")

        class BaseForestQuantileRegressor(base):
            def __init__(self,
                         n_estimators=10,
                         criterion='mse',
                         max_depth=None,
                         min_samples_split=2,
                         min_samples_leaf=1,
                         min_weight_fraction_leaf=0.0,
                         max_features='auto',
                         max_leaf_nodes=None,
                         bootstrap=True,
                         oob_score=False,
                         n_jobs=1,
                         random_state=None,
                         verbose=0,
                         warm_start=False):
                super(BaseForestQuantileRegressor, self).__init__(
                    base_estimator=base_estimator,
                    n_estimators=n_estimators,
                    estimator_params=("criterion", "max_depth", "min_samples_split",
                                      "min_samples_leaf", "min_weight_fraction_leaf",
                                      "max_features", "max_leaf_nodes",
                                      "random_state"),
                    bootstrap=bootstrap,
                    oob_score=oob_score,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbose=verbose,
                    warm_start=warm_start)

                self.criterion = criterion
                self.max_depth = max_depth
                self.min_samples_split = min_samples_split
                self.min_samples_leaf = min_samples_leaf
                self.min_weight_fraction_leaf = min_weight_fraction_leaf
                self.max_features = max_features
                self.max_leaf_nodes = max_leaf_nodes
                self.method = method

            def __repr__(self):
                s = super(BaseForestQuantileRegressor, self).__repr__()
                params = f"{s[s.find('('):-1]}, method='{method}')"
                if type == "random_forest":
                    class_name = 'RandomForestQuantileRegressor'
                elif type == "extra_trees":
                    class_name = 'ExtraTreesQuantileRegressor'
                return class_name + params

        return BaseForestQuantileRegressor(**kwargs)


class RandomForestQuantileRegressor:
    """
    A random forest regressor that provides quantile estimates.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
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

    max_depth : integer or None, optional (default=None)
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
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    method : str, ['default', 'sample', 'tdigest']
        Method for the calculations. 'default' uses the method outlined in
        the original paper. 'sample' uses the approach as currently used
        in the R package quantRegForest. 'tdigest' stores information on
        the distribution for each leaf and combines this information at
        quantile calculation. 'default' has the highest precision but
        is slower, 'tdigest' is an intermediate and 'sample' is relatively
        fast (but needs most memory as samples of y are stored in each
        estimator). Depending on the method additional attributes are stored
        in the model.

        Default:
            y_train_ : array-like, shape=(n_samples,)
                Cache the target values at fit time.

            y_weights_ : array-like, shape=(n_estimators, n_samples)
                y_weights_[i, j] is the weight given to sample ``j` while
                estimator ``i`` is fit. If bootstrap is set to True, this
                reduces to a 2-D array of ones.

            y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
                y_train_leaves_[i, j] provides the leaf node that y_train_[i]
                ends up when estimator j is fit. If y_train_[i] is given
                a weight of zero when estimator j is fit, then the value is -1.

        Sample:
            each estimator contains

            training_data_ : pd.DataFrame
                The dataframe index contains the prediction leaf indices. One
                column per n_outputs contains the sampled observation

        TDigest:
            each estimator contains

            training_data_ : pd.DataFrame
                The dataframe index contains the prediction leaf indices. One
                column per n_outputs contains a TDigest object with information
                on the distribution of the training observations in that leaf.

    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """

    def __new__(cls, *, method='default', **kwargs):
        return _ForestQuantileRegressor(method=method, type='random_forest', **kwargs)


class ExtraTreesQuantileRegressor:
    """
    An extra-trees regressor that provides quantile estimates.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
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

    max_depth : integer or None, optional (default=None)
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
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of ExtraTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = (n_features)
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = (n_samples)
        Prediction computed with out-of-bag estimate on the training set.

    method : str, ['default', 'sample', 'tdigest']
        Method for the calculations. 'default' uses the method outlined in
        the original paper. 'sample' uses the approach as currently used
        in the R package quantRegForest. 'tdigest' stores information on
        the distribution for each leaf and combines this information at
        quantile calculation. 'default' has the highest precision but
        is slower, 'tdigest' is an intermediate and 'sample' is relatively
        fast (but needs most memory as samples of y are stored in each
        estimator). Depending on the method additional attributes are stored
        in the model.

        Default:
            y_train_ : array-like, shape=(n_samples,)
                Cache the target values at fit time.

            y_weights_ : array-like, shape=(n_estimators, n_samples)
                y_weights_[i, j] is the weight given to sample ``j` while
                estimator ``i`` is fit. If bootstrap is set to True, this
                reduces to a 2-D array of ones.

            y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
                y_train_leaves_[i, j] provides the leaf node that y_train_[i]
                ends up when estimator j is fit. If y_train_[i] is given
                a weight of zero when estimator j is fit, then the value is -1.

        Sample:
            each estimator contains

            training_data_ : pd.DataFrame
                The dataframe index contains the prediction leaf indices. One
                column per n_outputs contains the sampled observation

        TDigest:
            each estimator contains

            training_data_ : pd.DataFrame
                The dataframe index contains the prediction leaf indices. One
                column per n_outputs contains a TDigest object with information
                on the distribution of the training observations in that leaf.

    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """

    def __new__(cls, *, method='default', **kwargs):
        return _ForestQuantileRegressor(method=method, type='extra_trees', **kwargs)


fit_docstring = \
"""
Build a forest from the training set (X, y).

Parameters
----------
X : array-like or sparse matrix, shape = (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    The target values

sample_weight : array-like, shape = (n_samples) or None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. Splits are also
    ignored if they would result in any single class carrying a
    negative weight in either child node.

Returns
-------
self : object
    Returns self.    
"""

predict_docstring = \
"""
Predict quantile regression values for X.

Parameters
----------
X : array-like or sparse matrix of shape = (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

q : float, optional
    Value ranging from 0 to 1. By default, the mean is returned.

Returns
-------
y : array of shape = (n_samples) or (n_samples, n_outputs)
    If quantile is set to None, then return E(Y | X). Else return
    y such that F(Y=y | x) = quantile.
"""

for Regressor in [_DefaultForestQuantileRegressor,
                  _RandomSampleForestQuantileRegressor,
                  _TDigestForestQuantileRegressor]:
    Regressor.fit.__doc__ = fit_docstring
    Regressor.predict.__doc__ = predict_docstring
