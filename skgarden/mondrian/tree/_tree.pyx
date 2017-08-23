# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from cpython cimport Py_INCREF, PyObject

from libc.math cimport exp
from libc.math cimport sqrt
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.string cimport memcpy
from libc.string cimport memset

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csr_matrix

from ._utils cimport Stack
from ._utils cimport StackRecord
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray
from ._utils cimport rand_exponential
from ._utils cimport rand_multinomial
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples', 'tau', 'variance'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64, np.float32, np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).tau,
        <Py_ssize_t> &(<Node*> NULL).variance,
    ]
})

cdef inline double fmax(double left, double right) nogil:
    return left if left > right else right


cdef inline double fmin(double left, double right) nogil:
    return left if left < right else right


# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

cdef class PartialFitTreeBuilder(TreeBuilder):
    """Build a decision tree incrementally."""

    def __cinit__(self, SIZE_t min_samples_split, SIZE_t max_depth,
                  object random_state):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.random_state = random_state

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        X, y, sample_weight = self._check_input(X, y, None)

        cdef UINT32_t rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef int n_samples = X.shape[0]

        # Allocate memory for tree.
        cdef int init_capacity
        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
            tree._resize(init_capacity)

        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef DOUBLE_t* y_ptr = <DOUBLE_t*> y.data
        cdef SIZE_t X_f_stride = X.strides[1] / X.itemsize
        cdef SIZE_t X_s_stride = X.strides[0] / X.itemsize
        cdef SIZE_t y_stride = y.strides[0] / y.itemsize
        cdef SIZE_t sample_ind
        cdef SIZE_t start

        # Initialize the tree when the first sample is inserted.
        if tree.node_count == 0:
            tree._init(X_ptr, y_ptr, X_f_stride)
            start = 1
        else:
            start = 0

        for sample_ind in range(start, n_samples):
            tree.extend(X_ptr, y_ptr, sample_ind*X_s_stride,
                        X_f_stride,
                        sample_ind*y_stride, rand_r_state,
                        self.min_samples_split)

# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t max_depth):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity
        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047
        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_split = self.min_samples_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double threshold
        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        with nogil:
            # push root node onto stack
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split)

                if not is_leaf:
                    is_leaf = splitter.node_split(impurity, &split, &n_constant_features)
                    is_leaf = is_leaf or (split.pos >= end)
                else:
                    splitter.set_bounds()

                # Check if the node is pure.
                is_leaf = is_leaf or splitter.criterion.is_pure()

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         weighted_n_node_samples,
                                         splitter.lower_bounds,
                                         splitter.upper_bounds,
                                         split.E)

                if node_id == <SIZE_t>(-1):
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:
                    # Push right child on stack
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()

# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The maximal depth of the tree.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    property tau:
        def __get__(self):
            return self._get_node_ndarray()["tau"][:self.node_count]

    property mean:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count].ravel()

    property variance:
        def __get__(self):
            return self._get_node_ndarray()["variance"][:self.node_count]

    property root:
        def __get__(self):
            return self.root

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_outputs):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

        # Used only in partial_fit
        self.root = 0

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is infered during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        d["root"] = self.root
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]
        self.root = d["root"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    # XXX using (size_t)(-1) is ugly, but SIZE_MAX is not available in C89
    # (i.e., older MSVC).
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil except -1:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef void update_node_extent(self, SIZE_t node_ind, SIZE_t child_ind,
                                 DTYPE_t* X_ptr, SIZE_t X_start, SIZE_t X_f_stride):
        """
        Updates the lower_bound and given_bound of the node at node_ind.
        The lower bound is the minimum of the lower bound at child_ind
        and the value of X_ptr.
        """
        cdef SIZE_t f_ind
        cdef DTYPE_t x_val
        cdef Node* node = &self.nodes[node_ind]
        cdef Node* prev_node = &self.nodes[child_ind]

        for f_ind in range(self.n_features):
            x_val = X_ptr[X_start + X_f_stride*f_ind]
            node.lower_bounds[f_ind] = fmin(x_val, prev_node.lower_bounds[f_ind])
            node.upper_bounds[f_ind] = fmax(x_val, prev_node.upper_bounds[f_ind])

    cdef void _update_node_info(self, SIZE_t parent_id, SIZE_t child_id,
                                DOUBLE_t* y_ptr, SIZE_t y_start):
        """
        Update the value at node parent_ind given the values at node
        child_id and y_ptr[y_start]
        """
        cdef SIZE_t is_regression = self.n_classes[0] == 1
        cdef SIZE_t child_ptr = child_id*self.value_stride
        cdef SIZE_t parent_ptr = parent_id*self.value_stride
        cdef SIZE_t c_ind
        cdef Node* child = &self.nodes[child_id]
        cdef Node* parent = &self.nodes[parent_id]
        cdef DTYPE_t new_sum
        cdef DTYPE_t old_mean
        cdef DTYPE_t new_mean
        cdef DTYPE_t ss

        if is_regression:
            # Update mean
            old_mean = self.value[child_ptr]
            new_sum = old_mean*child.n_node_samples + y_ptr[y_start]
            new_mean = new_sum / (child.n_node_samples + 1)
            self.value[parent_ptr] = new_mean

            # Update variance
            ss = (child.variance + old_mean**2)*child.n_node_samples
            parent.variance = (
                (ss + y_ptr[y_start]**2) / (child.n_node_samples + 1) -
                new_mean**2)
        else:
            # Update class counts.
            self.value[parent_ptr + <SIZE_t> y_ptr[y_start]] += 1.0

            if child_id != parent_id:
                for c_ind in range(self.n_classes[0]):
                    self.value[parent_ptr + c_ind] += self.value[child_ptr + c_ind]

    cdef void set_node_attributes(self, SIZE_t node_ind, SIZE_t left_child,
                                  SIZE_t right_child, SIZE_t feature, DOUBLE_t threshold,
                                  DTYPE_t tau, SIZE_t n_node_samples,
                                  DOUBLE_t weighted_n_node_samples, DOUBLE_t impurity,
                                  DOUBLE_t variance, SIZE_t X_start,
                                  SIZE_t X_f_stride, DTYPE_t* X_ptr,
                                  DOUBLE_t* y_ptr, SIZE_t child_ind=-1,
                                  SIZE_t y_start=0):
        """
        Sets the left_child, right_child, feature, threshold, time of split,
        number of samples, impurity, variance of the node at node_ind.

        If child_ind is not provided, the node at node_ind is assumed to be a
        leaf node with X_ptr[X_start: X_start+n_features*X_f_stride]

        If child ind is provided, the node at node_ind is assumed to be
        the parent of the node at child_ind and the leaf node with
        X_ptr[X_start: X_start + n_features*X_f_stride]
        """
        cdef Node* node = &self.nodes[node_ind]
        cdef Node* prev_node
        cdef DTYPE_t x_val
        cdef SIZE_t f_ind
        cdef SIZE_t val_ptr = node_ind*self.value_stride

        node.left_child = left_child
        node.right_child = right_child
        node.feature = feature
        node.threshold = threshold
        node.tau = tau
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        node.impurity = impurity
        node.variance = variance
        node.lower_bounds = <DTYPE_t*> malloc(self.n_features * sizeof(DTYPE_t))
        node.upper_bounds = <DTYPE_t*> malloc(self.n_features * sizeof(DTYPE_t))

        # Set bounds.
        # If child_ind is -1, its a leaf, else update the extent of each node.
        if child_ind == -1:
            for f_ind in range(self.n_features):
                x_val = X_ptr[X_start + X_f_stride*f_ind]
                node.lower_bounds[f_ind] = node.upper_bounds[f_ind] = x_val
        else:
            self.update_node_extent(
                node_ind, child_ind, X_ptr, X_start, X_f_stride)

        # Set value at node_ind
        if child_ind == -1:
            # Regression
            if self.n_classes[0] == 1:
                self.value[val_ptr] = y_ptr[y_start]
            else:
                self.value[val_ptr + <SIZE_t> y_ptr[y_start]] = 1.0

    cdef void _init(self, DTYPE_t* X_ptr, DOUBLE_t* y_ptr, SIZE_t X_f_stride):
        """
        Parameters
        ----------
        X_ptr: DTYPE_t*, pointer to X

        y_ptr: DTYPE_t* pointer to y

        X_f_stride: SIZE_t, stride to reach consecutive feature.
        """
        self.set_node_attributes(0, _TREE_LEAF, _TREE_LEAF, _TREE_UNDEFINED,
                                 _TREE_UNDEFINED, INFINITY, 1, 1, 0.0, 0.0,
                                 0, X_f_stride, X_ptr, y_ptr)
        self.node_count += 1

    cdef void extend(self, DTYPE_t* X_ptr, DOUBLE_t* y_ptr, SIZE_t X_start,
                     SIZE_t X_f_stride, SIZE_t y_start, UINT32_t random_state,
                     SIZE_t min_samples_split):
        """
        Extends the tree given a new sample.
        (X_ptr[X_start: X_start+ n_features*X_f_stride], y_ptr[y_start])

        References
        ----------
        1. Algorithm 5.5, Decision Trees and Forests: A Probabilistic Perspective,
           Balaji Lakshminarayanan
           http://www.gatsby.ucl.ac.uk/~balaji/balaji-phd-thesis.pdf
        """
        # Traverse the tree
        cdef SIZE_t curr_id = self.root
        cdef SIZE_t parent_id = -1
        cdef SIZE_t left_id
        cdef SIZE_t right_id
        cdef SIZE_t new_child_id
        cdef SIZE_t new_parent_id
        cdef SIZE_t f_ind
        cdef SIZE_t feature
        cdef SIZE_t delta

        cdef Node* curr_node
        cdef Node* parent_node
        cdef Node* node

        cdef DTYPE_t x
        cdef DTYPE_t x_val
        cdef DTYPE_t new_rate
        cdef DTYPE_t* e_l = <DTYPE_t*> malloc(self.n_features * sizeof(DTYPE_t))
        cdef DTYPE_t* e_u = <DTYPE_t*> malloc(self.n_features * sizeof(DTYPE_t))
        cdef DTYPE_t* extent = <DTYPE_t*> malloc(self.n_features * sizeof(DTYPE_t))
        cdef DTYPE_t E
        cdef DTYPE_t tau_parent = 0.0
        cdef DTYPE_t threshold
        cdef DTYPE_t l_b
        cdef DTYPE_t u_b
        cdef int c_ind
        cdef SIZE_t rc

        while True:
            curr_node = &self.nodes[curr_id]

            # Step 1: Calculate e^l, e^u and rate.
            # If x belongs to the bounding box, this is zero.
            new_rate = 0.0
            for f_ind in range(self.n_features):
                x = X_ptr[X_start + f_ind*X_f_stride]
                e_l[f_ind] = fmax(curr_node.lower_bounds[f_ind] - x, 0)
                e_u[f_ind] = fmax(x - curr_node.upper_bounds[f_ind], 0)
                extent[f_ind] = e_l[f_ind] + e_u[f_ind]
                new_rate += extent[f_ind]

            # Step 2: Sample E from an exponential distribution.
            E = rand_exponential(new_rate, &random_state)

            # Step 3: Induce split.
            # 2 new nodes are created.
            # 1. A child node with the new sample.
            # 2. A parent node with the new child node and the node at
            # curr_id as children.
            if (tau_parent + E < curr_node.tau and
                curr_node.n_node_samples + 1 >= min_samples_split):

                new_child_id = self.node_count
                new_parent_id = self.node_count + 1

                # Step 4: Sample delta from a multinomial.
                delta = rand_multinomial(extent, self.n_features, &random_state)

                # Step 5: Sample xi uniformly between bounds.
                x_val = X_ptr[X_start + delta * X_f_stride]
                l_b = curr_node.lower_bounds[delta]
                u_b = curr_node.upper_bounds[delta]
                if x_val > u_b:
                    xi = rand_uniform(u_b, x_val, &random_state)
                else:
                    xi = rand_uniform(x_val, l_b, &random_state)

                # Step 7: Split criteria.
                if x_val < xi:
                    left_child = new_child_id
                    right_child = curr_id
                else:
                    left_child = curr_id
                    right_child = new_child_id

                # Allocate memory for the new parent and child.
                # Store leaf in nodes[self.node_count]
                # Store parent in nodes[self.node_count + 1]
                rc = self._resize_c(self.node_count + 2)
                if rc == -1:
                    raise MemoryError()

                # xxx: We need to get the pointer to curr_id again
                # because of the resizing above.
                curr_node = &self.nodes[curr_id]

                # Step 7-8: Create new leaf node j'' and update value.
                self.set_node_attributes(
                    new_child_id, _TREE_LEAF, _TREE_LEAF, _TREE_UNDEFINED,
                    _TREE_UNDEFINED, INFINITY, 1, 1, 0.0, 0.0, X_start,
                    X_f_stride, X_ptr, y_ptr, -1, y_start)

                # Step 6 : Create new parent node j'
                self.set_node_attributes(
                    new_parent_id, left_child, right_child, delta, xi,
                    tau_parent + E, curr_node.n_node_samples + 1,
                    curr_node.weighted_n_node_samples + 1, 0.0, 0.0, X_start,
                    X_f_stride, X_ptr, y_ptr, curr_id)
                self._update_node_info(new_parent_id, curr_id, y_ptr, y_start)

                # New root if curr_id is root.
                if curr_id == self.root:
                    self.root = new_parent_id
                else:
                    # Link to the newly created node j' (new_parent_id)
                    # as the child of the parent of node j (curr_id)
                    parent_node = &self.nodes[parent_id]
                    if parent_node.left_child == curr_id:
                        parent_node.left_child = new_parent_id
                    else:
                        parent_node.right_child = new_parent_id
                self.max_depth += 1
                self.node_count += 2
                break

            # Absorb new sample into curr_id and Traverse further down the
            # tree.
            else:
                # Step 10: Update extent, value at node curr_id and increment
                # the number of samples.
                self.update_node_extent(curr_id, curr_id, X_ptr, X_start, X_f_stride)
                self._update_node_info(curr_id, curr_id, y_ptr, y_start)
                curr_node.n_node_samples += 1
                curr_node.weighted_n_node_samples += 1

                if curr_node.left_child == -1:
                    break

                # Step 12 - 13: Recurse down the tree.
                parent_id = curr_id
                if X_ptr[X_start + curr_node.feature*X_f_stride] < curr_node.threshold:
                    curr_id = curr_node.left_child
                else:
                    curr_id = curr_node.right_child
                tau_parent = curr_node.tau
        free(e_l)
        free(e_u)
        free(extent)

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_node_samples,
                          DTYPE_t* lower_bounds,
                          DTYPE_t* upper_bounds,
                          double E) nogil except -1:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if parent == _TREE_UNDEFINED:
            node.tau = E
        elif is_leaf:
            node.tau = INFINITY
        else:
            node.tau = E + self.nodes[parent].tau

        node.lower_bounds = <DTYPE_t*> malloc(self.n_features * sizeof(DTYPE_t))
        node.upper_bounds = <DTYPE_t*> malloc(self.n_features * sizeof(DTYPE_t))
        memcpy(node.lower_bounds, lower_bounds, self.n_features*sizeof(DTYPE_t))
        memcpy(node.upper_bounds, upper_bounds, self.n_features*sizeof(DTYPE_t))
        node.variance = impurity

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED
        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold
        self.node_count += 1
        return node_id

    cpdef np.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        return self._apply_dense(X)

    cpdef tuple predict(self, object X, bint return_std=False, bint is_regression=True):
        """Predicts the regressor and standard deviation for all samples."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t f_ind

        # We currently support only single-output y.
        # These node values are the means in case of regression.
        # For classification these are the class counts.
        cdef np.ndarray[DOUBLE_t, ndim=2] node_values = self._get_value_ndarray()[:, 0, :]

        # Initialize output
        cdef np.ndarray[DTYPE_t, ndim=1] mean = np.zeros(n_samples, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] std = np.zeros(n_samples, dtype=DTYPE)

        cdef SIZE_t n_classes = node_values.shape[1]
        cdef np.ndarray[DTYPE_t, ndim=2] proba = np.zeros((n_samples, n_classes), dtype=DTYPE)
        cdef np.ndarray[SIZE_t, ndim=1] n_node_samples = self.n_node_samples

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t j
        cdef SIZE_t node_id = 0
        cdef DOUBLE_t Delta = 0.0
        cdef DOUBLE_t parent_tau
        cdef DOUBLE_t p_js
        cdef DOUBLE_t X_val
        cdef DOUBLE_t w_j

        # Algorithm 6.5
        cdef DOUBLE_t p_nsy
        cdef SIZE_t sample_ind
        cdef SIZE_t class_ind

        with nogil:
            for i in range(n_samples):
                # Step 3
                parent_tau = 0.0
                p_nsy = 1.0
                node_id = self.root

                while True:
                    node = &self.nodes[node_id]

                    # Step 5: First part.
                    # Calculate Delta
                    Delta = node.tau - parent_tau
                    parent_tau = node.tau

                    # Step 5: Second part.
                    # Calculate eta
                    eta = 0.0
                    for f_ind in range(n_features):
                        X_val = X_ptr[X_sample_stride*i + X_fx_stride*f_ind]

                        eta += (fmax(X_val - node.upper_bounds[f_ind], 0) +
                                fmax(node.lower_bounds[f_ind] - X_val, 0))

                    # Step 6: Calculate p_j
                    # Step 7-11
                    if node.left_child == _TREE_LEAF:
                        w_j = p_nsy
                    else:
                        p_js = 1 - exp(-Delta * eta)
                        w_j = p_nsy * p_js

                    if is_regression:
                        mean[i] += w_j * node_values[node_id, 0]
                    else:
                        for class_ind in range(n_classes):
                            proba[i, class_ind] += w_j * (node_values[node_id, class_ind] / n_node_samples[node_id])

                    if return_std:
                        std[i] += w_j * (node_values[node_id, 0]**2 + node.variance)

                    if node.left_child == _TREE_LEAF:
                        break
                    p_nsy = p_nsy * (1 - p_js)

                    # Step 12-14
                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        node_id = node.left_child
                    else:
                        node_id = node.right_child

                if return_std:
                    std[i] -= mean[i]**2
                    if std[i] <= 0:
                        std[i] = 0.0
                    std[i] = sqrt(std[i])

        if is_regression:
            if return_std:
                return mean, std
            return mean,
        else:
            return proba,


    cdef inline np.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t f_ind

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t curr_node_id

        with nogil:
            for i in range(n_samples):
                curr_node_id = self.root

                while True:

                    node = &self.nodes[curr_node_id]

                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:

                        if node.left_child == _TREE_LEAF:
                            break
                        curr_node_id = node.left_child

                    else:
                        if node.right_child == _TREE_LEAF:
                            break
                        curr_node_id = node.right_child
                out_ptr[i] = curr_node_id
        return out

    cpdef object decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""
        return self._decision_path_dense(X)

    cpdef object weighted_decision_path(self, object X):
        """Returns the weight at each node for each sample in X."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        cdef np.ndarray[DTYPE_t] values = np.zeros(n_samples *
                                                (1 + self.max_depth),
                                                 dtype=DTYPE)
        cdef DTYPE_t* values_ptr = <DTYPE_t*> values.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        cdef DTYPE_t parent_tau
        cdef DTYPE_t delta
        cdef DTYPE_t eta
        cdef DTYPE_t X_val
        cdef DTYPE_t p_s
        cdef DTYPE_t p_nsy
        cdef SIZE_t f_ind
        cdef SIZE_t curr_node_id

        with nogil:
            for i in range(n_samples):
                p_nsy = 1.0
                parent_tau = 0.0
                indptr_ptr[i + 1] = indptr_ptr[i]

                curr_node_id = self.root
                node = &self.nodes[curr_node_id]

                while node.left_child != _TREE_LEAF:

                    delta = node.tau - parent_tau
                    parent_tau = node.tau

                    eta = 0.0
                    for f_ind in range(n_features):
                        X_val = X_ptr[X_sample_stride * i + X_fx_stride * f_ind]
                        eta += (fmax(X_val - node.upper_bounds[f_ind], 0.0) +
                                fmax(node.lower_bounds[f_ind] - X_val, 0.0))
                    p_s = 1 - exp(-delta*eta)

                    if p_s > 0:
                        indices_ptr[indptr_ptr[i + 1]] = curr_node_id
                        values_ptr[indptr_ptr[i + 1]] = p_s * p_nsy
                        indptr_ptr[i + 1] += 1

                    p_nsy *= (1 - p_s)
                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        curr_node_id = node.left_child
                    else:
                        curr_node_id = node.right_child
                    node = &self.nodes[curr_node_id]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = curr_node_id
                values_ptr[indptr_ptr[i + 1]] = p_nsy
                indptr_ptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        values = values[:indptr[n_samples]]
        out = csr_matrix((values, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef inline object _decision_path_dense(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t curr_node_id

        with nogil:
            for i in range(n_samples):
                curr_node_id = self.root
                indptr_ptr[i + 1] = indptr_ptr[i]

                # Add all external nodes
                while curr_node_id != _TREE_LEAF:
                    node = &self.nodes[curr_node_id]

                    # ... and node.right_child != _TREE_LEAF:
                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        curr_node_id = node.left_child
                    else:
                        curr_node_id = node.right_child

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
