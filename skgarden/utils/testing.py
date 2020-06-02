"""
This module contains assertion function removed in scikit-learn 0.23
"""

# Authors: Florian Seidel (seidel.florian@gmail.com)
#
# License: BSD 3 clause

import unittest
from unittest import TestCase

from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_approx_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_less


__all__ = ["assert_equal", "assert_not_equal", "assert_raises",
           "assert_almost_equal", "assert_array_equal",
           "assert_array_almost_equal", "assert_array_less",
           "assert_less", "assert_less_equal",
           "assert_greater", "assert_greater_equal",
           "assert_approx_equal", "assert_allclose",
           "SkipTest", "assert_false", "assert_true"]

_dummy = TestCase('__init__')
assert_true = _dummy.assertTrue
assert_false = _dummy.assertFalse
assert_equal = _dummy.assertEqual
assert_not_equal = _dummy.assertNotEqual
assert_raises = _dummy.assertRaises
SkipTest = unittest.case.SkipTest
assert_dict_equal = _dummy.assertDictEqual
assert_in = _dummy.assertIn
assert_not_in = _dummy.assertNotIn
assert_less = _dummy.assertLess
assert_greater = _dummy.assertGreater
assert_less_equal = _dummy.assertLessEqual
assert_greater_equal = _dummy.assertGreaterEqual