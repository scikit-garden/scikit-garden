# Scikit-garden

Scikit-garden or skgarden (pronounced as skarden) is a garden for scikit-learn compatible trees.

## Installation

Scikit-Garden can be installed using pip.

```
pip install scikit-garden
```

## Usage

The estimators in Scikit-Garden are Scikit-Learn compatible and can serve as a drop-in replacement for Scikit-Learn's trees and forests.

```python
from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target

### Use MondrianForests for variance estimation
from skgarden import MondrianForestRegressor
mfr = MondrianForestRegressor()
mfr.fit(X, y)
y_mean, y_std = mfr.predict(X, return_std=True)

### Use QuantileForests for quantile estimation
from skgarden import RandomForestQuantileRegressor
rfqr = RandomForestQuantileRegressor(random_state=0)
rfqr.fit(X, y)
y_mean = rfqr.predict(X)
y_median = rfqr.predict(X, 50)
```
