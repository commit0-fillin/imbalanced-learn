"""Class to perform under-sampling by generating centroids based on
clustering."""
import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from ...utils import Substitution
from ...utils._docstring import _random_state_docstring
from ...utils._param_validation import HasMethods, StrOptions
from ..base import BaseUnderSampler
VOTING_KIND = ('auto', 'hard', 'soft')

@Substitution(sampling_strategy=BaseUnderSampler._sampling_strategy_docstring, random_state=_random_state_docstring)
class ClusterCentroids(BaseUnderSampler):
    """Undersample by generating centroids based on clustering methods.

    Method that under samples the majority class by replacing a
    cluster of majority samples by the cluster centroid of a KMeans
    algorithm.  This algorithm keeps N majority samples by fitting the
    KMeans algorithm with N cluster to the majority class and using
    the coordinates of the N cluster centroids as the new majority
    samples.

    Read more in the :ref:`User Guide <cluster_centroids>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    estimator : estimator object, default=None
        A scikit-learn compatible clustering method that exposes a `n_clusters`
        parameter and a `cluster_centers_` fitted attribute. By default, it will
        be a default :class:`~sklearn.cluster.KMeans` estimator.

    voting : {{"hard", "soft", "auto"}}, default='auto'
        Voting strategy to generate the new samples:

        - If ``'hard'``, the nearest-neighbors of the centroids found using the
          clustering algorithm will be used.
        - If ``'soft'``, the centroids found by the clustering algorithm will
          be used.
        - If ``'auto'``, if the input is sparse, it will default on ``'hard'``
          otherwise, ``'soft'`` will be used.

        .. versionadded:: 0.3.0

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    estimator_ : estimator object
        The validated estimator created from the `estimator` parameter.

    voting_ : str
        The validated voting strategy.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    EditedNearestNeighbours : Under-sampling by editing samples.

    CondensedNearestNeighbour: Under-sampling by condensing samples.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> from imblearn.under_sampling import ClusterCentroids
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> cc = ClusterCentroids(
    ...     estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42
    ... )
    >>> X_res, y_res = cc.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{...}})
    """
    _parameter_constraints: dict = {**BaseUnderSampler._parameter_constraints, 'estimator': [HasMethods(['fit', 'predict']), None], 'voting': [StrOptions({'auto', 'hard', 'soft'})], 'random_state': ['random_state']}

    def __init__(self, *, sampling_strategy='auto', random_state=None, estimator=None, voting='auto'):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.estimator = estimator
        self.voting = voting

    def _validate_estimator(self):
        """Private function to create the KMeans estimator"""
        pass