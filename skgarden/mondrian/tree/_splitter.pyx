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
#
# License: BSD 3 clause

from ._criterion cimport Criterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.stdlib cimport malloc
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport log as ln

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csc_matrix

from ._utils cimport log
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc

cdef double INFINITY = np.inf

cdef class Splitter:
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(self, Criterion criterion, object random_state):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.random_state = random_state

    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self,
                   object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight,
                   np.ndarray X_idx_sorted=None) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : numpy.ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples

        sample_weight : numpy.ndarray, dtype=DOUBLE_t (optional)
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize

        self.sample_weight = sample_weight
        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : numpy.ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.y_stride,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()

    cdef void set_bounds(self) nogil:
        pass

cdef class BaseDenseSplitter(Splitter):
    cdef DTYPE_t* X
    cdef SIZE_t X_sample_stride
    cdef SIZE_t X_feature_stride

    cdef np.ndarray X_idx_sorted
    cdef INT32_t* X_idx_sorted_ptr
    cdef SIZE_t X_idx_sorted_stride
    cdef SIZE_t n_total_samples
    cdef SIZE_t* sample_mask

    def __cinit__(self, Criterion criterion, object random_state):

        self.X = NULL
        self.X_sample_stride = 0
        self.X_feature_stride = 0
        self.X_idx_sorted_ptr = NULL
        self.X_idx_sorted_stride = 0
        self.sample_mask = NULL

    cdef int init(self,
                  object X,
                  np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                  DOUBLE_t* sample_weight,
                  np.ndarray X_idx_sorted=None) except -1:
        """Initialize the splitter

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        # Call parent init
        Splitter.init(self, X, y, sample_weight)

        # Initialize X
        cdef np.ndarray X_ndarray = X

        self.X = <DTYPE_t*> X_ndarray.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        self.X_feature_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        return 0


cdef class MondrianSplitter(BaseDenseSplitter):
    """Splitter that samples a tree from a mondrian process."""

    def __dealloc__(self):
        free(self.lower_bounds)
        free(self.upper_bounds)

    def __reduce__(self):
        return (MondrianSplitter, (self.criterion,
                                   self.random_state), self.__getstate__())

    cdef void set_bounds(self) nogil:
        """Sets lower bounds and upper bounds of every feature."""
        cdef SIZE_t n_features = self.n_features

        safe_realloc(&self.lower_bounds, n_features)
        safe_realloc(&self.upper_bounds, n_features)
        cdef DTYPE_t upper_bound
        cdef DTYPE_t lower_bound
        cdef DTYPE_t* X = self.X

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_feature_stride = self.X_feature_stride
        cdef SIZE_t f_j
        cdef DTYPE_t current_f
        cdef SIZE_t p
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        for f_j in range(n_features):
            upper_bound = -INFINITY
            lower_bound = INFINITY

            for p in range(start, end):
                current_f = X[samples[p]*X_sample_stride + f_j*X_feature_stride]
                if current_f <= lower_bound:
                    lower_bound = current_f
                if current_f > upper_bound:
                    upper_bound = current_f
            self.upper_bounds[f_j] = upper_bound
            self.lower_bounds[f_j] = lower_bound

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the mondrian split on node samples[start:end]

        Both the split feature and split threshold are determined independently
        of the labels.

        1. The upper bounds u_j and lower bounds l_j of all features in a
           given node j are determined.
        2. The split feature is drawn with a probability proportional to
           u_j - l_j.
        3. After choosing the split feature, the split location is drawn
           uniformly between the upper and lower bound of the split feature.

        In addition to the split feature and threshold, the time of split
        tau is also stored which is sampled from an exponential with rate
        equal to the sum of the difference across all dimensions.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        References
        ----------
        * Balaji Lakshminarayanan,
          Decision Trees and Forests: A probabilistic perspective.
          Pg 82, Algorithm 6.1 and 6.2
          http://www.gatsby.ucl.ac.uk/~balaji/balaji-phd-thesis.pdf
        """
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values

        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_feature_stride = self.X_feature_stride
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t tmp
        cdef SIZE_t feature_stride
        cdef SIZE_t partition_end
        cdef DTYPE_t rate = 0.0
        cdef DTYPE_t upper_bound
        cdef DTYPE_t lower_bound
        cdef DTYPE_t* cum_diff = <DTYPE_t*> malloc(n_features * sizeof(DTYPE_t))
        cdef DTYPE_t search

        self.set_bounds()
        # Sample E from sum(u_{d} - l_{d})
        for f_j in range(n_features):
            upper_bound = self.upper_bounds[f_j]
            lower_bound = self.lower_bounds[f_j]
            cum_diff[f_j] = upper_bound - lower_bound

            if f_j != 0:
                cum_diff[f_j] += cum_diff[f_j - 1]
            rate += (upper_bound - lower_bound)

        # Sample time of split to be -ln(U) / rate.
        split.E = -ln(rand_uniform(0.0, 1.0, random_state)) / rate

        # Sample dimension delta with a probability proportional to (u_d - l_d)
        search = rand_uniform(0.0, cum_diff[n_features-1], random_state)
        for f_j in range(n_features):
            if f_j == 0:
                lower_bound = 0.0
            else:
                lower_bound = cum_diff[f_j - 1]
            if cum_diff[f_j] >= search and lower_bound < search:
                split.feature = f_j
                break

        # Sample location xi uniformly between (l_d[delta], u_d[delta])
        split.threshold = rand_uniform(
            self.lower_bounds[split.feature],
            self.upper_bounds[split.feature],
            random_state)

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        feature_stride = X_feature_stride * split.feature
        partition_end = end
        p = start
        while p < partition_end:
            if X[X_sample_stride * samples[p] + feature_stride] <= split.threshold:
                p += 1
            else:
                partition_end -= 1
                tmp = samples[partition_end]
                samples[partition_end] = samples[p]
                samples[p] = tmp

        split.pos = p
        self.criterion.reset()
        self.criterion.update(split.pos)
        self.criterion.children_impurity(&split.impurity_left,
                                         &split.impurity_right)
        free(cum_diff)
        return 0
