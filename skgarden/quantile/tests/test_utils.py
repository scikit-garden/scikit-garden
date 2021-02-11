import numpy as np
from skgarden.quantile.utils import weighted_quantile

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
        [weighted_quantile(x, q, weights) for q in np.arange(0.1, 1.0, 0.1)]
    )
    assert_array_almost_equal(expected, actual)

    # check quantiles at (5, 95) at intervals of 10
    actual = (
        [weighted_quantile(x, q, weights) for q in np.arange(0.05, 1.05, 0.1)]
    )
    assert_array_almost_equal(sorted_x, actual)


def test_percentile_toy_data():
    x = [1, 2, 3]
    weights = [1, 4, 5]

    assert_equal(weighted_quantile(x, 0.0, weights), 1)
    assert_equal(weighted_quantile(x, 1.0, weights), 3)

    assert_equal(weighted_quantile(x, 0.05, weights), 1)
    assert_equal(weighted_quantile(x, 0.30, weights), 2)
    assert_equal(weighted_quantile(x, 0.75, weights), 3)
    assert_almost_equal(weighted_quantile(x, 0.50, weights), 2.44, 2)


def test_zero_weights():
    x = [1, 2, 3, 4, 5]
    w = [0, 0, 0, 0.1, 0.1]

    for q in np.arange(0.0, 1.10, 0.1):
        assert_equal(
            weighted_quantile(x, q, w),
            weighted_quantile([4, 5], q, [0.1, 0.1])
        )

if __name__ == "skgarden.quantile.tests.test_utils":
    print("Test utils")
    test_percentile_equal_weights()
    test_percentile_toy_data()
    test_zero_weights()
