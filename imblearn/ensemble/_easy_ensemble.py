"""Class to perform under-sampling using easy ensemble."""
import copy
import numbers
import warnings
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble._bagging import _parallel_decision_function
from sklearn.ensemble._base import _partition_estimators
from sklearn.exceptions import NotFittedError
from sklearn.utils._tags import _safe_tags
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_is_fitted
try:
    from sklearn.utils.parallel import Parallel, delayed
except (ImportError, ModuleNotFoundError):
    from joblib import Parallel
    from sklearn.utils.fixes import delayed
from ..base import _ParamsValidationMixin
from ..pipeline import Pipeline
from ..under_sampling import RandomUnderSampler
from ..under_sampling.base import BaseUnderSampler
from ..utils import Substitution, check_sampling_strategy, check_target_type
from ..utils._available_if import available_if
from ..utils._docstring import _n_jobs_docstring, _random_state_docstring
from ..utils._param_validation import Interval, StrOptions
from ..utils.fixes import _fit_context
from ._common import _bagging_parameter_constraints, _estimator_has
MAX_INT = np.iinfo(np.int32).max
sklearn_version = parse_version(sklearn.__version__)

@Substitution(sampling_strategy=BaseUnderSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring, random_state=_random_state_docstring)
class EasyEnsembleClassifier(_ParamsValidationMixin, BaggingClassifier):
    """Bag of balanced boosted learners also known as EasyEnsemble.

    This algorithm is known as EasyEnsemble [1]_. The classifier is an
    ensemble of AdaBoost learners trained on different balanced bootstrap
    samples. The balancing is achieved by random under-sampling.

    Read more in the :ref:`User Guide <boosting>`.

    .. versionadded:: 0.4

    Parameters
    ----------
    n_estimators : int, default=10
        Number of AdaBoost learners in the ensemble.

    estimator : estimator object, default=AdaBoostClassifier()
        The base AdaBoost classifier used in the inner ensemble. Note that you
        can set the number of inner learner by passing your own instance.

        .. versionadded:: 0.10

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.

    {sampling_strategy}

    replacement : bool, default=False
        Whether or not to sample randomly with replacement or not.

    {n_jobs}

    {random_state}

    verbose : int, default=0
        Controls the verbosity of the building process.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 0.10

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples for each base estimator.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : array, shape (n_classes,)
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    n_features_ : int
        The number of features when `fit` is performed.

        .. deprecated:: 1.0
           `n_features_` is deprecated in `scikit-learn` 1.0 and will be removed
           in version 1.2. When the minimum version of `scikit-learn` supported
           by `imbalanced-learn` will reach 1.2, this attribute will be removed.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.9

    See Also
    --------
    BalancedBaggingClassifier : Bagging classifier for which each base
        estimator is trained on a balanced bootstrap.

    BalancedRandomForestClassifier : Random forest applying random-under
        sampling to balance the different bootstraps.

    RUSBoostClassifier : AdaBoost classifier were each bootstrap is balanced
        using random-under sampling at each round of boosting.

    Notes
    -----
    The method is described in [1]_.

    Supports multi-class resampling by sampling each class independently.

    References
    ----------
    .. [1] X. Y. Liu, J. Wu and Z. H. Zhou, "Exploratory Undersampling for
       Class-Imbalance Learning," in IEEE Transactions on Systems, Man, and
       Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550,
       April 2009.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import confusion_matrix
    >>> from imblearn.ensemble import EasyEnsembleClassifier
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> eec = EasyEnsembleClassifier(random_state=42)
    >>> eec.fit(X_train, y_train)
    EasyEnsembleClassifier(...)
    >>> y_pred = eec.predict(X_test)
    >>> print(confusion_matrix(y_test, y_pred))
    [[ 23   0]
     [  2 225]]
    """
    if sklearn_version >= parse_version('1.4'):
        _parameter_constraints = copy.deepcopy(BaggingClassifier._parameter_constraints)
    else:
        _parameter_constraints = copy.deepcopy(_bagging_parameter_constraints)
    excluded_params = {'bootstrap', 'bootstrap_features', 'max_features', 'oob_score', 'max_samples'}
    for param in excluded_params:
        _parameter_constraints.pop(param, None)
    _parameter_constraints.update({'sampling_strategy': [Interval(numbers.Real, 0, 1, closed='right'), StrOptions({'auto', 'majority', 'not minority', 'not majority', 'all'}), dict, callable], 'replacement': ['boolean']})
    if 'base_estimator' in _parameter_constraints:
        del _parameter_constraints['base_estimator']

    def __init__(self, n_estimators=10, estimator=None, *, warm_start=False, sampling_strategy='auto', replacement=False, n_jobs=None, random_state=None, verbose=0):
        super().__init__(n_estimators=n_estimators, max_samples=1.0, max_features=1.0, bootstrap=False, bootstrap_features=False, oob_score=False, warm_start=warm_start, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
        self.estimator = estimator
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement

    def _validate_estimator(self, default=AdaBoostClassifier(algorithm='SAMME')):
        """Check the estimator and the n_estimator attribute, set the
        `estimator_` attribute."""
        pass

    @property
    def n_features_(self):
        """Number of features when ``fit`` is performed."""
        pass

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y):
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass

    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X):
        """Average of the decision functions of the base classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        score : ndarray of shape (n_samples, k)
            The decision function of the input samples. The columns correspond
            to the classes in sorted order, as they appear in the attribute
            ``classes_``. Regression and binary classification are special
            cases with ``k == 1``, otherwise ``k==n_classes``.
        """
        pass

    @property
    def base_estimator_(self):
        """Attribute for older sklearn version compatibility."""
        pass