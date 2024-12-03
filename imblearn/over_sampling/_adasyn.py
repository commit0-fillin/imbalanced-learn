"""Class to perform over-sampling using ADASYN."""
import numbers
import warnings
import numpy as np
from scipy import sparse
from sklearn.utils import _safe_indexing, check_random_state
from ..utils import Substitution, check_neighbors_object
from ..utils._docstring import _n_jobs_docstring, _random_state_docstring
from ..utils._param_validation import HasMethods, Interval
from .base import BaseOverSampler

@Substitution(sampling_strategy=BaseOverSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring, random_state=_random_state_docstring)
class ADASYN(BaseOverSampler):
    """Oversample using Adaptive Synthetic (ADASYN) algorithm.

    This method is similar to SMOTE but it generates different number of
    samples depending on an estimate of the local distribution of the class
    to be oversampled.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    n_neighbors : int or estimator object, default=5
        The nearest neighbors used to define the neighborhood of samples to use
        to generate the synthetic samples. You can pass:

        - an `int` corresponding to the number of neighbors to use. A
          `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this
          case.
        - an instance of a compatible nearest neighbors algorithm that should
          implement both methods `kneighbors` and `kneighbors_graph`. For
          instance, it could correspond to a
          :class:`~sklearn.neighbors.NearestNeighbors` but could be extended to
          any compatible class.

    {n_jobs}

        .. deprecated:: 0.10
           `n_jobs` has been deprecated in 0.10 and will be removed in 0.12.
           It was previously used to set `n_jobs` of nearest neighbors
           algorithm. From now on, you can pass an estimator where `n_jobs` is
           already set instead.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours estimator linked to the parameter `n_neighbors`.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    BorderlineSMOTE : Over-sample using Borderline-SMOTE variant.

    Notes
    -----
    The implementation is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used.

    References
    ----------
    .. [1] He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. "ADASYN:
       Adaptive synthetic sampling approach for imbalanced learning," In IEEE
       International Joint Conference on Neural Networks (IEEE World Congress
       on Computational Intelligence), pp. 1322-1328, 2008.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import ADASYN
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000,
    ... random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> ada = ADASYN(random_state=42)
    >>> X_res, y_res = ada.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 904, 1: 900}})
    """
    _parameter_constraints: dict = {**BaseOverSampler._parameter_constraints, 'n_neighbors': [Interval(numbers.Integral, 1, None, closed='left'), HasMethods(['kneighbors', 'kneighbors_graph'])], 'n_jobs': [numbers.Integral, None]}

    def __init__(self, *, sampling_strategy='auto', random_state=None, n_neighbors=5, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the necessary objects for ADASYN"""
        self.nn_ = check_neighbors_object(
            'n_neighbors', self.n_neighbors, additional_neighbor=1
        )
        if self.n_jobs is not None:
            warnings.warn(
                "The parameter `n_jobs` has been deprecated in 0.10 and will be "
                "removed in 0.12. To configure the number of jobs of the nearest "
                "neighbors estimator, use `ADASYN(n_neighbors="
                "NearestNeighbors(n_jobs=...))`.",
                FutureWarning,
            )
            self.nn_.set_params(n_jobs=self.n_jobs)
