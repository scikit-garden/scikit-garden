import numpy as np
from crick.tdigest import TDigest
from numba import jit, float32, float64, int64, prange
import numba as nb
from sklearn.utils import check_random_state


@jit(float32[:](float32[:, :], float32, float32[:]), nopython=True, cache=True)
def _weighted_quantile(a, q, weights):
    """
    Weighted quantile calculation.

    Parameters
    ----------
    a : array, shape = (n_sample, n_features)
        Data from which the quantiles are calculated. One quantile value
        per feature (n_features) is given. Should be float32.
    q : float
        Quantile in range [0, 1]. Should be a float32 value.
    weights : array, shape = (n_sample)
        Weights of each sample. Should be float32

    Returns
    -------
    quantiles : array, shape = (n_features)
        Quantile values

    References
    ----------
    1. https://en.wikipedia.org/wiki/Percentile#The_Weighted_Percentile_method

    Notes
    -----
    Note that weighted_quantile(a, q) is not equivalent to np.quantile(a, q).
    This is because in np.quantile sorted(a)[i] is assumed to be at quantile 0.0,
    while here we assume sorted(a)[i] is given a weight of 1.0 / len(a), hence
    it is at the 1.0 / len(a)th quantile.
    """
    nz = weights != 0
    a = a[nz]
    weights = weights[nz]

    n_features = a.shape[1]
    quantiles = np.full(n_features, np.nan, dtype=np.float32)
    if a.shape[0] == 1 or a.size == 0:
        return a[0]

    for i in range(n_features):
        sorted_indices = np.argsort(a[:, i])
        sorted_a = a[sorted_indices, i]
        sorted_weights = weights[sorted_indices]


        # Step 1
        sorted_cum_weights = np.cumsum(sorted_weights)
        total = sorted_cum_weights[-1]

        # Step 2
        partial_sum = 1 / total * (sorted_cum_weights - sorted_weights / 2.0)
        start = np.searchsorted(partial_sum, q) - 1
        if start == len(sorted_cum_weights) - 1:
            quantiles[i] = sorted_a[-1]
            continue
        if start == -1:
            quantiles[i] = sorted_a[0]
            continue

        # Step 3.
        fraction = (q - partial_sum[start]) / (partial_sum[start + 1] - partial_sum[start])
        quantiles[i] = sorted_a[start] + fraction * (sorted_a[start + 1] - sorted_a[start])
    return quantiles


def weighted_quantile(a, q, weights=None):
    """
    Returns the weighted quantile of a at q given weights.

    Parameters
    ----------
    a: array-like, shape=(n_samples, n_features)
        Samples from which the quantile is calculated

    q: float
        Quantile (in the range from 0-1)

    weights: array-like, shape=(n_samples,)
        Weights[i] is the weight given to point a[i] while computing the
        quantile. If weights[i] is zero, a[i] is simply ignored during the
        quantile computation.

    Returns
    -------
    quantile: array, shape = (n_features)
        Weighted quantile of a at q.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Percentile#The_Weighted_Percentile_method

    Notes
    -----
    Note that weighted_quantile(a, q) is not equivalent to np.quantile(a, q).
    This is because in np.quantile sorted(a)[i] is assumed to be at quantile 0.0,
    while here we assume sorted(a)[i] is given a weight of 1.0 / len(a), hence
    it is at the 1.0 / len(a)th quantile.
    """
    if q > 1 or q < 0:
        raise ValueError("q should be in-between 0 and 1, "
                         "got %d" % q)

    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape((-1, 1))
    elif a.ndim > 2:
        raise ValueError("a should be in the format (n_samples, n_feature)")

    if weights is None:
        weights = np.ones(a.shape[0], dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim > 1:
            raise ValueError("weights need to be 1 dimensional")

    if a.shape[0] != weights.shape[0]:
        raise ValueError("a and weights should have the same length.")

    q = np.float32(q)

    quantiles = _weighted_quantile(a, q, weights)

    if quantiles.size == 1:
        return quantiles[0]
    else:
        return quantiles


@jit(float32[:, :](int64[:], float32[:, :], int64[:], float32[:], float32), parallel=True, cache=True)
def _quantile_tree_predict(X_leaves, y_train, y_train_leaves, y_weights, q):
    quantiles = np.zeros((X_leaves.shape[0], y_train.shape[1]), dtype=np.float32)
    for i in prange(len(X_leaves)):
        mask = y_train_leaves == X_leaves[i]
        quantiles[i] = _weighted_quantile(y_train[mask], q, y_weights[mask])
    return quantiles


@jit(float32[:, :](int64[:, :], float32[:, :], int64[:, :], float32[:, :], float32), parallel=True, cache=True)
def _quantile_forest_predict(X_leaves, y_train, y_train_leaves, y_weights, q):
    quantiles = np.zeros((X_leaves.shape[0], y_train.shape[1]), dtype=np.float32)
    for i in prange(len(X_leaves)):
        x_leaf = X_leaves[i]
        x_weights = np.zeros(y_weights.shape[1], dtype=np.float32)
        for j in range(y_weights.shape[1]):
            x_weights[j] += (y_weights[:, j] * (y_train_leaves[:, j] == x_leaf)).sum()
        quantiles[i] = _weighted_quantile(y_train, q, x_weights)
    return quantiles


@jit(nopython=True, cache=True)
def rand_choice_nb(a, p):
    """
    Random choice from array with probabilities per sample

    Parameters
    ----------
    a : np.array
        Data
    p : float
        Probability of each observation. Should sum up to 1.

    Returns
    -------
    A random sample from the given array with a given probability.

    References
    ----------
    - https://github.com/numba/numba/issues/2539
    """
    return a[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]


@jit(nb.types.containers.UniTuple(int64[:], 2)(int64[:], float64[:], int64[:]), nopython=True, cache=True)
def _weighted_random_sample(leaves, weights, idx):
    """
    Random sample for each unique leaf

    Parameters
    ----------
    leaves : array, shape = (n_samples)
        Leaves of a Regression tree, corresponding to weights and indices (idx)
    weights : array, shape = (n_samples)
        Weights for each observation. They don't need to sum up to 1.
    idx : array, shape = (n_samples)
        Indices of original observations. The output will drawn from this.

    Returns
    -------
    unique_leaves, sampled_idx, shape = (n_unique_samples)
        Unique leaves (from 'leaves') and a randomly (and weighted) sample
        from 'idx' corresponding to the leaf.
    """
    unique_leaves = np.unique(leaves)
    sampled_idx = np.empty_like(unique_leaves, dtype=np.int64)
    for i in prange(len(unique_leaves)):
        mask = unique_leaves[i] == leaves
        c_weights = weights[mask]
        c_idx = idx[mask]
        sampled_idx[i] = rand_choice_nb(c_idx, p=c_weights/c_weights.sum())
    return unique_leaves, sampled_idx


def generate_sample_indices(random_state, n_samples):
    """
    Generates bootstrap indices for each tree fit.

    Parameters
    ----------
    random_state: int, RandomState instance or None
        If int, random_state is the seed used by the random number generator.
        If RandomState instance, random_state is the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    n_samples: int
        Number of samples to generate from each tree.

    Returns
    -------
    sample_indices: array-like, shape=(n_samples), dtype=np.int32
        Sample indices.
    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices


def set_tdigest(x, w):
    """
    Create TDigest object and insert weighted data

    Parameters
    ----------
    x : array, shape = (n_samples)
        Data to be inserted to the TDigest object
    w : array, shape = (n_samples)
        Weights per data point

    Returns
    -------
    TDigest
    """
    t = TDigest()
    t.update(x, w)
    return t


def tdigestlist_quantile(t_list, q):
    """
    Merge TDigest objects and calculate quantile

    Parameters
    ----------
    t_list : array-like
        TDigest objects that will be merged
    q : float
        Quantile to be calculated from the merged TDigest, range 0 to 1.

    Returns
    Quantile
    """
    t = TDigest()
    t.merge(*(t_list.values))
    return t.quantile(q)
