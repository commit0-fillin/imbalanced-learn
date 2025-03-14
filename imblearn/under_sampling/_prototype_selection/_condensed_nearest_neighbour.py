"""Class to perform under-sampling based on the condensed nearest neighbour
method."""
import numbers
import warnings
from collections import Counter
import numpy as np
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import _safe_indexing, check_random_state
from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring, _random_state_docstring
from ...utils._param_validation import HasMethods, Interval
from ..base import BaseCleaningSampler

@Substitution(sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring, random_state=_random_state_docstring)
class CondensedNearestNeighbour(BaseCleaningSampler):
    """Undersample based on the condensed nearest neighbour method.

    Read more in the :ref:`User Guide <condensed_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    n_neighbors : int or estimator object, default=None
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors.  If `None`, a
        :class:`~sklearn.neighbors.KNeighborsClassifier` with a 1-NN rules will
        be used.

    n_seeds_S : int, default=1
        Number of samples to extract in order to build the set S.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    estimator_ : estimator object
        The validated K-nearest neighbor estimator created from `n_neighbors` parameter.

        .. deprecated:: 0.12
           `estimator_` is deprecated in 0.12 and will be removed in 0.14. Use
           `estimators_` instead that contains the list of all K-nearest
           neighbors estimator used for each pair of class.

    estimators_ : list of estimator objects of shape (n_resampled_classes - 1,)
        Contains the K-nearest neighbor estimator used for per of classes.

        .. versionadded:: 0.12

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
    EditedNearestNeighbours : Undersample by editing samples.

    RepeatedEditedNearestNeighbours : Undersample by repeating ENN algorithm.

    AllKNN : Undersample using ENN and various number of neighbours.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling: a strategy one (minority) vs. each other
    classes is applied.

    References
    ----------
    .. [1] P. Hart, "The condensed nearest neighbor rule,"
       In Information Theory, IEEE Transactions on, vol. 14(3),
       pp. 515-516, 1968.

    Examples
    --------
    >>> from collections import Counter  # doctest: +SKIP
    >>> from sklearn.datasets import fetch_openml  # doctest: +SKIP
    >>> from sklearn.preprocessing import scale  # doctest: +SKIP
    >>> from imblearn.under_sampling import CondensedNearestNeighbour  # doctest: +SKIP
    >>> X, y = fetch_openml('diabetes', version=1, return_X_y=True)  # doctest: +SKIP
    >>> X = scale(X)  # doctest: +SKIP
    >>> print('Original dataset shape %s' % Counter(y))  # doctest: +SKIP
    Original dataset shape Counter({{'tested_negative': 500,         'tested_positive': 268}})  # doctest: +SKIP
    >>> cnn = CondensedNearestNeighbour(random_state=42)  # doctest: +SKIP
    >>> X_res, y_res = cnn.fit_resample(X, y)  #doctest: +SKIP
    >>> print('Resampled dataset shape %s' % Counter(y_res))  # doctest: +SKIP
    Resampled dataset shape Counter({{'tested_positive': 268,         'tested_negative': 181}})  # doctest: +SKIP
    """
    _parameter_constraints: dict = {**BaseCleaningSampler._parameter_constraints, 'n_neighbors': [Interval(numbers.Integral, 1, None, closed='left'), HasMethods(['kneighbors', 'kneighbors_graph']), None], 'n_seeds_S': [Interval(numbers.Integral, 1, None, closed='left')], 'n_jobs': [numbers.Integral, None], 'random_state': ['random_state']}

    def __init__(self, *, sampling_strategy='auto', random_state=None, n_neighbors=None, n_seeds_S=1, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        if self.n_neighbors is None:
            estimator = KNeighborsClassifier(n_neighbors=1, n_jobs=self.n_jobs)
        elif isinstance(self.n_neighbors, int):
            estimator = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs)
        else:
            estimator = clone(self.n_neighbors)

        self.estimator_ = estimator

    @property
    def estimator_(self):
        """Last fitted k-NN estimator."""
        return self._estimator if hasattr(self, '_estimator') else None

    @estimator_.setter
    def estimator_(self, estimator):
        self._estimator = estimator
