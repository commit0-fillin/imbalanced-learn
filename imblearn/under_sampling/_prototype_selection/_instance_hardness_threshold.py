"""Class to perform under-sampling based on the instance hardness
threshold."""
import numbers
from collections import Counter
import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils import _safe_indexing, check_random_state
from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring, _random_state_docstring
from ...utils._param_validation import HasMethods
from ..base import BaseUnderSampler

@Substitution(sampling_strategy=BaseUnderSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring, random_state=_random_state_docstring)
class InstanceHardnessThreshold(BaseUnderSampler):
    """Undersample based on the instance hardness threshold.

    Read more in the :ref:`User Guide <instance_hardness_threshold>`.

    Parameters
    ----------
    estimator : estimator object, default=None
        Classifier to be used to estimate instance hardness of the samples.
        This classifier should implement `predict_proba`.

    {sampling_strategy}

    {random_state}

    cv : int, default=5
        Number of folds to be used when estimating samples' instance hardness.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        correspond to the class labels from which to sample and the values
        are the number of samples to sample.

    estimator_ : estimator object
        The validated classifier used to estimate the instance hardness of the samples.

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
    NearMiss : Undersample based on near-miss search.

    RandomUnderSampler : Random under-sampling.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling: from each class to be under-sampled, it
    retains the observations with the highest probability of being correctly
    classified.

    References
    ----------
    .. [1] D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier.
       "An instance level analysis of data complexity." Machine learning
       95.2 (2014): 225-256.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import InstanceHardnessThreshold
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> iht = InstanceHardnessThreshold(random_state=42)
    >>> X_res, y_res = iht.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 5..., 0: 100}})
    """
    _parameter_constraints: dict = {**BaseUnderSampler._parameter_constraints, 'estimator': [HasMethods(['fit', 'predict_proba']), None], 'cv': ['cv_object'], 'n_jobs': [numbers.Integral, None], 'random_state': ['random_state']}

    def __init__(self, *, estimator=None, sampling_strategy='auto', random_state=None, cv=5, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs

    def _validate_estimator(self, random_state):
        """Private function to create the classifier"""
        if self.estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=10,
                random_state=random_state,
                n_jobs=self.n_jobs
            )
        else:
            estimator = clone(self.estimator)

        if isinstance(estimator, RandomForestClassifier):
            _set_random_states(estimator, random_state)

        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                f"{estimator.__class__.__name__} doesn't have predict_proba method."
            )

        return estimator
