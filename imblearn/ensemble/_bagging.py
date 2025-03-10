"""Bagging classifier trained on balanced bootstrap samples."""
import copy
import numbers
import warnings
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble._bagging import _parallel_decision_function
from sklearn.ensemble._base import _partition_estimators
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
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
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.fixes import _fit_context
from ._common import _bagging_parameter_constraints, _estimator_has
sklearn_version = parse_version(sklearn.__version__)

@Substitution(sampling_strategy=BaseUnderSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring, random_state=_random_state_docstring)
class BalancedBaggingClassifier(_ParamsValidationMixin, BaggingClassifier):
    """A Bagging classifier with additional balancing.

    This implementation of Bagging is similar to the scikit-learn
    implementation. It includes an additional step to balance the training set
    at fit time using a given sampler.

    This classifier can serves as a basis to implement various methods such as
    Exactly Balanced Bagging [6]_, Roughly Balanced Bagging [7]_,
    Over-Bagging [6]_, or SMOTE-Bagging [8]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    estimator : estimator object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

        .. versionadded:: 0.10

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator.

        - If int, then draw ``max_samples`` samples.
        - If float, then draw ``max_samples * X.shape[0]`` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

        - If int, then draw ``max_features`` features.
        - If float, then draw ``max_features * X.shape[1]`` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement.

        .. note::
           Note that this bootstrap will be generated from the resampled
           dataset.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.

    {sampling_strategy}

    replacement : bool, default=False
        Whether or not to randomly sample with replacement or not when
        `sampler is None`, corresponding to a
        :class:`~imblearn.under_sampling.RandomUnderSampler`.

    {n_jobs}

    {random_state}

    verbose : int, default=0
        Controls the verbosity of the building process.

    sampler : sampler object, default=None
        The sampler used to balanced the dataset before to bootstrap
        (if `bootstrap=True`) and `fit` a base estimator. By default, a
        :class:`~imblearn.under_sampling.RandomUnderSampler` is used.

        .. versionadded:: 0.8

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 0.10

    n_features_ : int
        The number of features when `fit` is performed.

        .. deprecated:: 1.0
           `n_features_` is deprecated in `scikit-learn` 1.0 and will be removed
           in version 1.2. When the minimum version of `scikit-learn` supported
           by `imbalanced-learn` will reach 1.2, this attribute will be removed.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    sampler_ : sampler object
        The validate sampler created from the `sampler` parameter.

    estimators_samples_ : list of ndarray
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.

    estimators_features_ : list of ndarray
        The subset of drawn features for each base estimator.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        ``oob_decision_function_`` might contain NaN.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.9

    See Also
    --------
    BalancedRandomForestClassifier : Random forest applying random-under
        sampling to balance the different bootstraps.

    EasyEnsembleClassifier : Ensemble of AdaBoost classifier trained on
        balanced bootstraps.

    RUSBoostClassifier : AdaBoost classifier were each bootstrap is balanced
        using random-under sampling at each round of boosting.

    Notes
    -----
    This is possible to turn this classifier into a balanced random forest [5]_
    by passing a :class:`~sklearn.tree.DecisionTreeClassifier` with
    `max_features='auto'` as a base estimator.

    See
    :ref:`sphx_glr_auto_examples_ensemble_plot_comparison_ensemble_classifier.py`.

    References
    ----------
    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.

    .. [5] C. Chen Chao, A. Liaw, and L. Breiman. "Using random forest to
           learn imbalanced data." University of California, Berkeley 110,
           2004.

    .. [6] R. Maclin, and D. Opitz. "An empirical evaluation of bagging and
           boosting." AAAI/IAAI 1997 (1997): 546-551.

    .. [7] S. Hido, H. Kashima, and Y. Takahashi. "Roughly balanced bagging
           for imbalanced data." Statistical Analysis and Data Mining: The ASA
           Data Science Journal 2.5‐6 (2009): 412-426.

    .. [8] S. Wang, and X. Yao. "Diversity analysis on imbalanced data sets by
           using ensemble models." 2009 IEEE symposium on computational
           intelligence and data mining. IEEE, 2009.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import confusion_matrix
    >>> from imblearn.ensemble import BalancedBaggingClassifier
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> bbc = BalancedBaggingClassifier(random_state=42)
    >>> bbc.fit(X_train, y_train)
    BalancedBaggingClassifier(...)
    >>> y_pred = bbc.predict(X_test)
    >>> print(confusion_matrix(y_test, y_pred))
    [[ 23   0]
     [  2 225]]
    """
    if sklearn_version >= parse_version('1.4'):
        _parameter_constraints = copy.deepcopy(BaggingClassifier._parameter_constraints)
    else:
        _parameter_constraints = copy.deepcopy(_bagging_parameter_constraints)
    _parameter_constraints.update({'sampling_strategy': [Interval(numbers.Real, 0, 1, closed='right'), StrOptions({'auto', 'majority', 'not minority', 'not majority', 'all'}), dict, callable], 'replacement': ['boolean'], 'sampler': [HasMethods(['fit_resample']), None]})
    if 'base_estimator' in _parameter_constraints:
        del _parameter_constraints['base_estimator']

    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, sampling_strategy='auto', replacement=False, n_jobs=None, random_state=None, verbose=0, sampler=None):
        super().__init__(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, bootstrap=bootstrap, bootstrap_features=bootstrap_features, oob_score=oob_score, warm_start=warm_start, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
        self.estimator = estimator
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        self.sampler = sampler

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `estimator_` attribute."""
        if self.estimator is not None:
            self.estimator_ = clone(self.estimator)
        else:
            self.estimator_ = clone(default)

        if isinstance(self.estimator_, ClassifierMixin):
            self._estimator_type = "classifier"
        else:
            raise ValueError(
                f"Estimator {self.estimator_} must be a ClassifierMixin."
            )

    @property
    def n_features_(self):
        """Number of features when ``fit`` is performed."""
        # Check if the estimator is fitted
        check_is_fitted(self)
        return self.estimators_[0].n_features_in_

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
        # Check that X and y have correct shape
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc'])

        # Check parameters
        self._validate_estimator()

        # Convert y to categorical if needed
        y = self._validate_y(y)

        # Check parameters
        self._validate_parameters()

        # Remap output
        n_samples, self.n_features_in_ = X.shape
        self._n_samples = n_samples
        y = np.copy(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Check sampling strategy
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        # Validate max_samples
        max_samples = self._validate_max_samples(n_samples)

        # Create sampler object
        if self.sampler is None:
            self.sampler_ = RandomUnderSampler(
                sampling_strategy=self.sampling_strategy_,
                replacement=self.replacement,
                random_state=self.random_state,
            )
        else:
            self.sampler_ = clone(self.sampler)

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )

        # Avoid storing the output of every estimator by summing them here
        if self.oob_score:
            self.estimators_samples_ = np.zeros((n_samples, self.n_estimators),
                                                dtype=np.int8)

        # Build estimators
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                self.max_samples,
                self.max_features,
                self.bootstrap,
                self.bootstrap_features,
                self.oob_score,
                self.sampling_strategy_,
                self.replacement,
                self.random_state,
                self.verbose,
                support_sample_weight=False,
                seeds=None,
                sampler=self.sampler_,
                estimator_params=self.estimator_params,
            )
            for i, _ in enumerate(starts)
        )

        # Reduce
        self.estimators_ = list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.n_features_ = self.estimators_[0].n_features_in_

        if self.oob_score:
            self.oob_score_ = self._compute_oob_score(X, y)

        return self

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
        check_is_fitted(self)
        
        # Check data
        X = self._validate_data(
            X, accept_sparse=['csr', 'csc'], reset=False,
            dtype=None, force_all_finite=False
        )

        # Parallel loop
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        all_decisions = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_decision_function)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X
            )
            for i in range(n_jobs)
        )

        # Reduce
        decisions = sum(all_decisions) / self.n_estimators

        return decisions

    @property
    def base_estimator_(self):
        """Attribute for older sklearn version compatibility."""
        warnings.warn(
            "Attribute `base_estimator_` was deprecated in version 0.24 and "
            "will be removed in 0.26. Use `estimator_` instead.",
            FutureWarning
        )
        return self.estimator_
