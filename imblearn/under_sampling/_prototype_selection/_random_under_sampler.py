"""Class to perform random under-sampling."""
import numpy as np
from sklearn.utils import _safe_indexing, check_random_state
from ...utils import Substitution, check_target_type
from ...utils._docstring import _random_state_docstring
from ...utils._validation import _check_X
from ..base import BaseUnderSampler

@Substitution(sampling_strategy=BaseUnderSampler._sampling_strategy_docstring, random_state=_random_state_docstring)
class RandomUnderSampler(BaseUnderSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    replacement : bool, default=False
        Whether the sample is with or without replacement.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

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
    NearMiss : Undersample using near-miss samples.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """
    _parameter_constraints: dict = {**BaseUnderSampler._parameter_constraints, 'replacement': ['boolean'], 'random_state': ['random_state']}

    def __init__(self, *, sampling_strategy='auto', random_state=None, replacement=False):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.replacement = replacement