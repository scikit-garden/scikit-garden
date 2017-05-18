# Differences from sklearn's tree-architecture

The mondrian forest implementation heavily relies on sklearn's tree documentation. Here is how the core differs from scikit-learn's implementation.

## Splitter
1. Added `MondrianSplitter`.
2. Added the field E to the `SplitRecord` struct to store the time of split.
3. Added the method `set_bounds` on the attributes `lower_bounds` and `upper_bounds` of the Splitter class.

## Criterion
The Criterion objects are just proxies to compute statistics, which are available from `sum_total`, `sum_left` and `sum_right` and play no real role in split computation.

## Node
The extra fields in the Node object are

1. lower_bounds
2. upper_bounds
3. mean
4. variance
5. tau

## Tree
The `tree_` attribute also returns the "tau", "mean" and "variance" of nodes through the `tau`, `mean` and `variance` attributes.
