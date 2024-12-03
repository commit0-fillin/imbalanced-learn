"""Classes to perform under-sampling based on the edited nearest neighbour
method."""
import numbers
from collections import Counter
import numpy as np
from sklearn.utils import _safe_indexing
from ...utils import Substitution, check_neighbors_object
from ...utils._validation import _deprecate_positional_args
from ...utils._docstring import _n_jobs_docstring
from ...utils._param_validation import HasMethods, Interval, StrOptions
from ...utils.fixes import _mode
from ..base import BaseCleaningSampler
SEL_KIND = ('all', 'mode')

@Substitution(sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring)
class EditedNearestNeighbours(BaseCleaningSampler):
    """Undersample based on the edited nearest neighbour method.

    This method cleans the dataset by removing samples close to the
    decision boundary. It removes observations from the majority class or
    classes when any or most of its closest neighours are from a different class.

    Read more in the :ref:`User Guide <edited_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    n_neighbors : int or object, default=3
        If ``int``, size of the neighbourhood to consider for the undersampling, i.e.,
        if `n_neighbors=3`, a sample will be removed when any or most of its 3 closest
        neighbours are from a different class. If object, an estimator that inherits
        from :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors. Note that if you want to examine the 3 closest
        neighbours of a sample for the undersampling, you need to pass a 4-KNN.

    kind_sel : {{'all', 'mode'}}, default='all'
        Strategy to use to exclude samples.

        - If ``'all'``, all neighbours should be of the same class of the examined
          sample for it not be excluded.
        - If ``'mode'``, most neighbours should be of the same class of the examined
          sample for it not be excluded.

        The strategy `"all"` will be less conservative than `'mode'`. Thus,
        more samples will be removed when `kind_sel="all"`, generally.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        correspond to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours instance created from `n_neighbors` parameter.

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
    CondensedNearestNeighbour : Undersample by condensing samples.

    RepeatedEditedNearestNeighbours : Undersample by repeating the ENN algorithm.

    AllKNN : Undersample using ENN with varying neighbours.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    References
    ----------
    .. [1] D. Wilson, Asymptotic" Properties of Nearest Neighbor Rules Using
       Edited Data," In IEEE Transactions on Systems, Man, and Cybernetrics,
       vol. 2 (3), pp. 408-421, 1972.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import EditedNearestNeighbours
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> enn = EditedNearestNeighbours()
    >>> X_res, y_res = enn.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 887, 0: 100}})
    """
    _parameter_constraints: dict = {**BaseCleaningSampler._parameter_constraints, 'n_neighbors': [Interval(numbers.Integral, 1, None, closed='left'), HasMethods(['kneighbors', 'kneighbors_graph'])], 'kind_sel': [StrOptions({'all', 'mode'})], 'n_jobs': [numbers.Integral, None]}

    def __init__(self, *, sampling_strategy='auto', n_neighbors=3, kind_sel='all', n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Validate the estimator created in the ENN."""
        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors)
        self.nn_.set_params(**{'n_neighbors': self.n_neighbors + 1})

        # Ensure n_neighbors is at least 2
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1.")

        # Set n_jobs parameter if provided
        if self.n_jobs is not None:
            self.nn_.set_params(**{'n_jobs': self.n_jobs})

@Substitution(sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring)
class RepeatedEditedNearestNeighbours(BaseCleaningSampler):
    """Undersample based on the repeated edited nearest neighbour method.

    This method repeats the :class:`EditedNearestNeighbours` algorithm several times.
    The repetitions will stop when i) the maximum number of iterations is reached,
    or ii) no more observations are being removed, or iii) one of the majority classes
    becomes a minority class or iv) one of the majority classes disappears
    during undersampling.

    Read more in the :ref:`User Guide <edited_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    n_neighbors : int or object, default=3
        If ``int``, size of the neighbourhood to consider for the undersampling, i.e.,
        if `n_neighbors=3`, a sample will be removed when any or most of its 3 closest
        neighbours are from a different class. If object, an estimator that inherits
        from :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors. Note that if you want to examine the 3 closest
        neighbours of a sample for the undersampling, you need to pass a 4-KNN.

    max_iter : int, default=100
        Maximum number of iterations of the edited nearest neighbours.

    kind_sel : {{'all', 'mode'}}, default='all'
        Strategy to use to exclude samples.

        - If ``'all'``, all neighbours should be of the same class of the examined
          sample for it not be excluded.
        - If ``'mode'``, most neighbours should be of the same class of the examined
          sample for it not be excluded.

        The strategy `"all"` will be less conservative than `'mode'`. Thus,
        more samples will be removed when `kind_sel="all"`, generally.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        correspond to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours estimator linked to the parameter `n_neighbors`.

    enn_ : sampler object
        The validated :class:`~imblearn.under_sampling.EditedNearestNeighbours`
        instance.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    n_iter_ : int
        Number of iterations run.

        .. versionadded:: 0.6

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    CondensedNearestNeighbour : Undersample by condensing samples.

    EditedNearestNeighbours : Undersample by editing samples.

    AllKNN : Undersample using ENN with varying neighbours.

    Notes
    -----
    The method is based on [1]_. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    Supports multi-class resampling.

    References
    ----------
    .. [1] I. Tomek, "An Experiment with the Edited Nearest-Neighbor
       Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6),
       pp. 448-452, June 1976.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> renn = RepeatedEditedNearestNeighbours()
    >>> X_res, y_res = renn.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 887, 0: 100}})
    """
    _parameter_constraints: dict = {**BaseCleaningSampler._parameter_constraints, 'n_neighbors': [Interval(numbers.Integral, 1, None, closed='left'), HasMethods(['kneighbors', 'kneighbors_graph'])], 'max_iter': [Interval(numbers.Integral, 1, None, closed='left')], 'kind_sel': [StrOptions({'all', 'mode'})], 'n_jobs': [numbers.Integral, None]}

    def __init__(self, *, sampling_strategy='auto', n_neighbors=3, max_iter=100, kind_sel='all', n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors)
        self.nn_.set_params(**{'n_neighbors': self.n_neighbors + 1})
        self.enn_ = EditedNearestNeighbours(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            kind_sel=self.kind_sel,
            n_jobs=self.n_jobs,
        )

@Substitution(sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring)
class AllKNN(BaseCleaningSampler):
    """Undersample based on the AllKNN method.

    This method will apply :class:`EditedNearestNeighbours` several times varying the
    number of nearest neighbours at each round. It begins by examining 1 closest
    neighbour, and it incrases the neighbourhood by 1 at each round.

    The algorithm stops when the maximum number of neighbours are examined or
    when the majority class becomes the minority class, whichever comes first.

    Read more in the :ref:`User Guide <edited_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    n_neighbors : int or estimator object, default=3
        If ``int``, size of the maximum neighbourhood to examine for the undersampling.
        If `n_neighbors=3`, in the first iteration the algorithm will examine 1 closest
        neigbhour, in the second round 2, and in the final round 3. If object, an
        estimator that inherits from :class:`~sklearn.neighbors.base.KNeighborsMixin`
        that will be used to find the nearest-neighbors. Note that if you want to
        examine the 3 closest neighbours of a sample, you need to pass a 4-KNN.

    kind_sel : {{'all', 'mode'}}, default='all'
        Strategy to use to exclude samples.

        - If ``'all'``, all neighbours should be of the same class of the examined
          sample for it not be excluded.
        - If ``'mode'``, most neighbours should be of the same class of the examined
          sample for it not be excluded.

        The strategy `"all"` will be less conservative than `'mode'`. Thus,
        more samples will be removed when `kind_sel="all"`, generally.

    allow_minority : bool, default=False
        If ``True``, it allows the majority classes to become the minority
        class without early stopping.

        .. versionadded:: 0.3

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        correspond to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours estimator linked to the parameter `n_neighbors`.

    enn_ : sampler object
        The validated :class:`~imblearn.under_sampling.EditedNearestNeighbours`
        instance.

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
    CondensedNearestNeighbour: Under-sampling by condensing samples.

    EditedNearestNeighbours: Under-sampling by editing samples.

    RepeatedEditedNearestNeighbours: Under-sampling by repeating ENN.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    References
    ----------
    .. [1] I. Tomek, "An Experiment with the Edited Nearest-Neighbor
       Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6),
       pp. 448-452, June 1976.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import AllKNN
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> allknn = AllKNN()
    >>> X_res, y_res = allknn.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 887, 0: 100}})
    """
    _parameter_constraints: dict = {**BaseCleaningSampler._parameter_constraints, 'n_neighbors': [Interval(numbers.Integral, 1, None, closed='left'), HasMethods(['kneighbors', 'kneighbors_graph'])], 'kind_sel': [StrOptions({'all', 'mode'})], 'allow_minority': ['boolean'], 'n_jobs': [numbers.Integral, None]}

    def __init__(self, *, sampling_strategy='auto', n_neighbors=3, kind_sel='all', allow_minority=False, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.allow_minority = allow_minority
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create objects required by AllKNN"""
        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors)
        self.nn_.set_params(**{'n_neighbors': self.n_neighbors})
        self.enn_ = EditedNearestNeighbours(
            sampling_strategy='all',
            n_neighbors=self.n_neighbors,
            kind_sel=self.kind_sel,
            n_jobs=self.n_jobs,
        )
