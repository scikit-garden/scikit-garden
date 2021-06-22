import numpy as np
from skgarden.quantile.utils import weighted_percentile

from numpy.testing import assert_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal

def test_percentile_equal_weights():
    rng = np.random.RandomState(0)
    x = rng.randn(10)
    weights = 0.1 * np.ones(10)

    # since weights are equal, quantiles lie in the midpoint.
    sorted_x = np.sort(x)
    expected = 0.5 * (sorted_x[1:] + sorted_x[:-1])
    actual = (
        [weighted_percentile(x, q, weights) for q in np.arange(10, 100, 10)]
    )
    assert_array_almost_equal(expected, actual)

    # check quantiles at (5, 95) at intervals of 10
    actual = (
        [weighted_percentile(x, q, weights) for q in np.arange(5, 105, 10)]
    )
    assert_array_almost_equal(sorted_x, actual)


def test_percentile_toy_data():
    x = [1, 2, 3]
    weights = [1, 4, 5]

    # Test 0 and 100th quantile
    assert_equal(weighted_percentile(x, 0, weights), 1)
    assert_equal(weighted_percentile(x, 100, weights), 3)

    assert_equal(weighted_percentile(x, 5, weights), 1)
    assert_equal(weighted_percentile(x, 30, weights), 2)
    assert_equal(weighted_percentile(x, 75, weights), 3)
    assert_almost_equal(weighted_percentile(x, 50, weights), 2.44, 2)


def test_zero_weights():
    x = [1, 2, 3, 4, 5]
    w = [0, 0, 0, 0.1, 0.1]

    for q in np.arange(0, 110, 10):
        assert_equal(
            weighted_percentile(x, q, w),
            weighted_percentile([4, 5], q, [0.1, 0.1])
        )
