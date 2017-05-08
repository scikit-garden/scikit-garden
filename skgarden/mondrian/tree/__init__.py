"""
The :mod:`sklearn.tree` module includes decision tree-based models for
classification and regression.
"""

from .tree import MondrianTreeRegressor
from .tree import MondrianTreeClassifier

__all__ = ["MondrianTreeClassifier", "MondrianTreeRegressor"]
