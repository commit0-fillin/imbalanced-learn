"""Class to perform under-sampling based on nearmiss methods."""
import numbers
import warnings
from collections import Counter
import numpy as np
from sklearn.utils import _safe_indexing
from ...utils import Substitution, check_neighbors_object
from ...utils._docstring import _n_jobs_docstring
from ...utils._param_validation import HasMethods, Interval
from ..base import BaseUnderSampler

@Substitution(sampling_strategy=BaseUnderSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring)
class NearMiss(BaseUnderSampler):
    """Class to perform under-sampling based on NearMiss methods.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

    Parameters
    ----------
    {sampling_strategy}

    version : int, default=1
        Version of the NearMiss to use. Possible values are 1, 2 or 3.

    n_neighbors : int or estimator object, default=3
        If ``int``, size of the neighbourhood to consider to compute the
        average distance to the minority point samples.  If object, an
        estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.
        By default, it will be a 3-NN.

    n_neighbors_ver3 : int or estimator object, default=3
        If ``int``, NearMiss-3 algorithm start by a phase of re-sampling. This
        parameter correspond to the number of neighbours selected create the
        subset in which the selection will be performed.  If object, an
        estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.
        By default, it will be a 3-NN.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours object created from `n_neighbors` parameter.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    RandomUnderSampler : Random undersample the dataset.

    InstanceHardnessThreshold : Use of classifier to undersample a dataset.

    Notes
    -----
    The methods are based on [1]_.

    Supports multi-class resampling.

    References
    ----------
    .. [1] I. Mani, I. Zhang. "kNN approach to unbalanced data distributions:
       a case study involving information extraction," In Proceedings of
       workshop on learning from imbalanced datasets, 2003.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import NearMiss
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> nm = NearMiss()
    >>> X_res, y_res = nm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """
    _parameter_constraints: dict = {**BaseUnderSampler._parameter_constraints, 'version': [Interval(numbers.Integral, 1, 3, closed='both')], 'n_neighbors': [Interval(numbers.Integral, 1, None, closed='left'), HasMethods(['kneighbors', 'kneighbors_graph'])], 'n_neighbors_ver3': [Interval(numbers.Integral, 1, None, closed='left'), HasMethods(['kneighbors', 'kneighbors_graph'])], 'n_jobs': [numbers.Integral, None]}

    def __init__(self, *, sampling_strategy='auto', version=1, n_neighbors=3, n_neighbors_ver3=3, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.version = version
        self.n_neighbors = n_neighbors
        self.n_neighbors_ver3 = n_neighbors_ver3
        self.n_jobs = n_jobs

    def _selection_dist_based(self, X, y, dist_vec, num_samples, key, sel_strategy='nearest'):
        """Select the appropriate samples depending of the strategy selected.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Original samples.

        y : array-like, shape (n_samples,)
            Associated label to X.

        dist_vec : ndarray, shape (n_samples, )
            The distance matrix to the nearest neighbour.

        num_samples: int
            The desired number of samples to select.

        key : str or int,
            The target class.

        sel_strategy : str, optional (default='nearest')
            Strategy to select the samples. Either 'nearest' or 'farthest'

        Returns
        -------
        idx_sel : ndarray, shape (num_samples,)
            The list of the indices of the selected samples.

        """
        # Find the indices of samples in the majority class
        idx_class = np.flatnonzero(y == key)

        # Sort the distances based on the strategy
        if sel_strategy == 'nearest':
            sort_way = False
        elif sel_strategy == 'farthest':
            sort_way = True
        else:
            raise ValueError("'sel_strategy' should be either 'nearest' or 'farthest'.")

        idx_sorted = idx_class[dist_vec[idx_class].argsort(kind='mergesort')]

        # Select the desired number of samples
        return idx_sorted[:num_samples]

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        from sklearn.neighbors import NearestNeighbors

        if isinstance(self.n_neighbors, int):
            self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs)
        elif isinstance(self.n_neighbors, NearestNeighbors):
            self.nn_ = self.n_neighbors
        else:
            raise ValueError(f"'n_neighbors' has to be a positive integer or a NearestNeighbors object. Got {type(self.n_neighbors)} instead.")

        if self.version == 3:
            if isinstance(self.n_neighbors_ver3, int):
                self.nn_ver3_ = NearestNeighbors(n_neighbors=self.n_neighbors_ver3, n_jobs=self.n_jobs)
            elif isinstance(self.n_neighbors_ver3, NearestNeighbors):
                self.nn_ver3_ = self.n_neighbors_ver3
            else:
                raise ValueError(f"'n_neighbors_ver3' has to be a positive integer or a NearestNeighbors object. Got {type(self.n_neighbors_ver3)} instead.")
