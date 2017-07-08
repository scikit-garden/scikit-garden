# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _tree.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    SIZE_t n_node_samples                # Number of samples at the node
    DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node
    DTYPE_t* lower_bounds                # Lower bounds of all features at current node.
    DTYPE_t* upper_bounds                # Upper bounds of all features at current node.
    DTYPE_t tau                          # Time of split.
    DOUBLE_t variance                    # Emprical variance of the node.


cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X
    cdef SIZE_t* n_classes               # Number of classes in y[:, k]
    cdef public SIZE_t n_outputs         # Number of outputs in y
    cdef public SIZE_t max_n_classes     # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
    cdef SIZE_t value_stride             # = n_outputs * max_n_classes

    cdef SIZE_t root
    # Methods
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_samples,
                          DTYPE_t* lower_bounds,
                          DTYPE_t* upper_bounds,
                          double E) nogil except -1
    cdef int _resize(self, SIZE_t capacity) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)

    cpdef tuple predict(self, object X, bint return_std=*, bint is_regression=*)

    cpdef np.ndarray apply(self, object X)
    cdef np.ndarray _apply_dense(self, object X)

    cpdef object decision_path(self, object X)
    cdef object _decision_path_dense(self, object X)
    cpdef object weighted_decision_path(self, object X)
    cdef void _init(self, DTYPE_t* X_ptr, DOUBLE_t* y_ptr, SIZE_t X_stride)
    cdef void extend(self, DTYPE_t* X_ptr, DOUBLE_t* y_ptr, SIZE_t x_start,
                     SIZE_t X_f_stride, SIZE_t y_stride, UINT32_t random_state,
                     SIZE_t min_samples_split)
    cdef void set_node_attributes(self, SIZE_t node_ind, SIZE_t left_child,
                                  SIZE_t right_child, SIZE_t feature, DOUBLE_t threshold,
                                  DTYPE_t tau, SIZE_t n_node_samples,
                                  DOUBLE_t weighted_n_node_samples, DOUBLE_t impurity,
                                  DOUBLE_t variance, SIZE_t X_start,
                                  SIZE_t X_f_stride, DTYPE_t* X_ptr,
                                  DOUBLE_t* y_ptr, SIZE_t child_ind=?,
                                  SIZE_t y_start=?)
    cdef void update_node_extent(self, SIZE_t node_ind, SIZE_t child_ind,
                                 DTYPE_t* X_ptr, SIZE_t X_start, SIZE_t X_f_stride)
    cdef void _update_node_info(self, SIZE_t parent_id, SIZE_t child_id,
                                DOUBLE_t* y_ptr, SIZE_t y_start)
# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef Splitter splitter          # Splitting algorithm

    cdef SIZE_t min_samples_split   # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf    # Minimum number of samples in a leaf
    cdef double min_weight_leaf     # Minimum weight in a leaf
    cdef SIZE_t max_depth           # Maximal tree depth
    cdef object random_state

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=*,
                np.ndarray X_idx_sorted=*)
    cdef _check_input(self, object X, np.ndarray y, np.ndarray sample_weight)
