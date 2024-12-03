"""Base class and original SMOTE methods for over-sampling"""
import math
import numbers
import warnings
import numpy as np
import sklearn
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import _safe_indexing, check_array, check_random_state
from sklearn.utils.fixes import parse_version
from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0
from sklearn.utils.validation import _num_features
from ...metrics.pairwise import ValueDifferenceMetric
from ...utils import Substitution, check_neighbors_object, check_target_type
from ...utils._docstring import _n_jobs_docstring, _random_state_docstring
from ...utils._param_validation import HasMethods, Interval, StrOptions
from ...utils._validation import _check_X
from ...utils.fixes import _is_pandas_df, _mode
from ..base import BaseOverSampler
sklearn_version = parse_version(sklearn.__version__).base_version
if parse_version(sklearn_version) < parse_version('1.5'):
    from sklearn.utils import _get_column_indices
else:
    from sklearn.utils._indexing import _get_column_indices

class BaseSMOTE(BaseOverSampler):
    """Base class for the different SMOTE algorithms."""
    _parameter_constraints: dict = {**BaseOverSampler._parameter_constraints, 'k_neighbors': [Interval(numbers.Integral, 1, None, closed='left'), HasMethods(['kneighbors', 'kneighbors_graph'])], 'n_jobs': [numbers.Integral, None]}

    def __init__(self, sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Check the NN estimators shared across the different SMOTE
        algorithms.
        """
        self.nn_k_ = check_neighbors_object('k_neighbors', self.k_neighbors)

    def _make_samples(self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, y=None):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        n_samples : int
            The number of samples to generate.

        step_size : float, default=1.0
            The step size to create samples.

        y : ndarray of shape (n_samples_all,), default=None
            The true target associated with `nn_data`. Used by Borderline SMOTE-2 to
            weight the distances in the sample generation process.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray of shape (n_samples_new,)
            Target values for synthetic samples.
        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)
        steps = step_size * random_state.uniform(size=n_samples)
        rows = samples_indices // nn_num.shape[1]
        cols = samples_indices % nn_num.shape[1]

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type, y)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)

        return X_new, y_new

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps, y_type=None, y=None):
        """Generate a synthetic sample.

        The rule for the generation is:

        .. math::
           \\mathbf{s_{s}} = \\mathbf{s_{i}} + \\mathcal{u}(0, 1) \\times
           (\\mathbf{s_{i}} - \\mathbf{s_{nn}}) \\,

        where \\mathbf{s_{s}} is the new synthetic samples, \\mathbf{s_{i}} is
        the current sample, \\mathbf{s_{nn}} is a randomly selected neighbors of
        \\mathbf{s_{i}} and \\mathcal{u}(0, 1) is a random number between [0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        rows : ndarray of shape (n_samples,), dtype=int
            Indices pointing at feature vector in X which will be used
            as a base for creating new samples.

        cols : ndarray of shape (n_samples,), dtype=int
            Indices pointing at which nearest neighbor of base feature vector
            will be used when creating new samples.

        steps : ndarray of shape (n_samples,), dtype=float
            Step sizes for new samples.

        y_type : str, int or None, default=None
            Class label of the current target classes for which we want to generate
            samples.

        y : ndarray of shape (n_samples_all,), default=None
            The true target associated with `nn_data`. Used by Borderline SMOTE-2 to
            weight the distances in the sample generation process.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Synthetically generated samples.
        """
        if issparse(X):
            return self._generate_samples_sparse(X, nn_data, nn_num, rows, cols, steps)
        else:
            return self._generate_samples_dense(X, nn_data, nn_num, rows, cols, steps)

    def _generate_samples_dense(self, X, nn_data, nn_num, rows, cols, steps):
        X_new = np.zeros((steps.shape[0], X.shape[1]))
        for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
            X_new[i] = X[row] - step * (X[row] - nn_data[nn_num[row, col]])
        return X_new

    def _generate_samples_sparse(self, X, nn_data, nn_num, rows, cols, steps):
        n_samples, n_features = X.shape
        rows = rows.astype(int)
        cols = cols.astype(int)

        data = X.data.copy()
        indptr = X.indptr.copy()
        indices = X.indices.copy()

        new_data = []
        new_indices = []
        new_indptr = [0]

        for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
            start = indptr[row]
            end = indptr[row + 1]
            
            # Find the features that are non-zero in either the current sample or its neighbor
            current_indices = indices[start:end]
            nn_start = nn_data.indptr[nn_num[row, col]]
            nn_end = nn_data.indptr[nn_num[row, col] + 1]
            nn_indices = nn_data.indices[nn_start:nn_end]
            all_indices = np.union1d(current_indices, nn_indices)
            
            for idx in all_indices:
                if idx in current_indices:
                    current_val = data[start + np.where(current_indices == idx)[0][0]]
                else:
                    current_val = 0
                
                if idx in nn_indices:
                    nn_val = nn_data.data[nn_start + np.where(nn_indices == idx)[0][0]]
                else:
                    nn_val = 0
                
                new_val = current_val - step * (current_val - nn_val)
                if new_val != 0:
                    new_data.append(new_val)
                    new_indices.append(idx)
            
            new_indptr.append(len(new_indices))

        return csr_matrix((new_data, new_indices, new_indptr), shape=(steps.shape[0], n_features))

    def _in_danger_noise(self, nn_estimator, samples, target_class, y, kind='danger'):
        """Estimate if a set of sample are in danger or noise.

        Used by BorderlineSMOTE and SVMSMOTE.

        Parameters
        ----------
        nn_estimator : estimator object
            An estimator that inherits from
            :class:`~sklearn.neighbors.base.KNeighborsMixin` use to determine
            if a sample is in danger/noise.

        samples : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        target_class : int or str
            The target corresponding class being over-sampled.

        y : array-like of shape (n_samples,)
            The true label in order to check the neighbour labels.

        kind : {'danger', 'noise'}, default='danger'
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray of shape (n_samples,)
            A boolean array where True refer to samples in danger or noise.
        """
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)

        if kind == 'danger':
            # Samples are in danger if there are more than 50% of majority samples in their neighborhood
            return np.ravel(n_maj > (nn_estimator.n_neighbors - 1) // 2)
        elif kind == 'noise':
            # Samples are noise if all neighbors are from the majority class
            return np.ravel(n_maj == nn_estimator.n_neighbors - 1)
        else:
            raise ValueError("'kind' should be either 'danger' or 'noise'.")

@Substitution(sampling_strategy=BaseOverSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring, random_state=_random_state_docstring)
class SMOTE(BaseSMOTE):
    """Class to perform over-sampling using SMOTE.

    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique as presented in [1]_.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
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

    nn_k_ : estimator object
        Validated k-nearest neighbours created from the `k_neighbors` parameter.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

    SVMSMOTE : Over-sample using the SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original papers: [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import SMOTE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> sm = SMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    def __init__(self, *, sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors, n_jobs=n_jobs)

@Substitution(sampling_strategy=BaseOverSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring, random_state=_random_state_docstring)
class SMOTENC(SMOTE):
    """Synthetic Minority Over-sampling Technique for Nominal and Continuous.

    Unlike :class:`SMOTE`, SMOTE-NC for dataset containing numerical and
    categorical features. However, it is not designed to work with only
    categorical features.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    .. versionadded:: 0.4

    Parameters
    ----------
    categorical_features : "infer" or array-like of shape (n_cat_features,) or             (n_features,), dtype={{bool, int, str}}
        Specified which features are categorical. Can either be:

        - "auto" (default) to automatically detect categorical features. Only
          supported when `X` is a :class:`pandas.DataFrame` and it corresponds
          to columns that have a :class:`pandas.CategoricalDtype`;
        - array of `int` corresponding to the indices specifying the categorical
          features;
        - array of `str` corresponding to the feature names. `X` should be a pandas
          :class:`pandas.DataFrame` in this case.
        - mask array of shape (n_features, ) and ``bool`` dtype for which
          ``True`` indicates the categorical features.

    categorical_encoder : estimator, default=None
        One-hot encoder used to encode the categorical features. If `None`, a
        :class:`~sklearn.preprocessing.OneHotEncoder` is used with default parameters
        apart from `handle_unknown` which is set to 'ignore'.

    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
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

    nn_k_ : estimator object
        Validated k-nearest neighbours created from the `k_neighbors` parameter.

    ohe_ : :class:`~sklearn.preprocessing.OneHotEncoder`
        The one-hot encoder used to encode the categorical features.

        .. deprecated:: 0.11
           `ohe_` is deprecated in 0.11 and will be removed in 0.13. Use
           `categorical_encoder_` instead.

    categorical_encoder_ : estimator
        The encoder used to encode the categorical features.

    categorical_features_ : ndarray of shape (n_cat_features,), dtype=np.int64
        Indices of the categorical features.

    continuous_features_ : ndarray of shape (n_cont_features,), dtype=np.int64
        Indices of the continuous features.

    median_std_ : dict of int -> float
        Median of the standard deviation of the continuous features for each
        class to be over-sampled.

    n_features_ : int
        Number of features observed at `fit`.

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

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    BorderlineSMOTE : Over-sample using Borderline-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original paper [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`,
    and
    :ref:`sphx_glr_auto_examples_over-sampling_plot_illustration_generation_sample.py`.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    Examples
    --------
    >>> from collections import Counter
    >>> from numpy.random import RandomState
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import SMOTENC
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print(f'Original dataset shape {{X.shape}}')
    Original dataset shape (1000, 20)
    >>> print(f'Original dataset samples per class {{Counter(y)}}')
    Original dataset samples per class Counter({{1: 900, 0: 100}})
    >>> # simulate the 2 last columns to be categorical features
    >>> X[:, -2:] = RandomState(10).randint(0, 4, size=(1000, 2))
    >>> sm = SMOTENC(random_state=42, categorical_features=[18, 19])
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print(f'Resampled dataset samples per class {{Counter(y_res)}}')
    Resampled dataset samples per class Counter({{0: 900, 1: 900}})
    """
    _required_parameters = ['categorical_features']
    _parameter_constraints: dict = {**SMOTE._parameter_constraints, 'categorical_features': ['array-like', StrOptions({'auto'})], 'categorical_encoder': [HasMethods(['fit_transform', 'inverse_transform']), None]}

    def __init__(self, categorical_features, *, categorical_encoder=None, sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors, n_jobs=n_jobs)
        self.categorical_features = categorical_features
        self.categorical_encoder = categorical_encoder
        self._validate_categorical_features()
        
    def _validate_categorical_features(self):
        if isinstance(self.categorical_features, str) and self.categorical_features != "auto":
            raise ValueError("categorical_features must be 'auto', array-like or a boolean mask")
        elif isinstance(self.categorical_features, (list, tuple, np.ndarray)):
            self.categorical_features = np.array(self.categorical_features)
        elif isinstance(self.categorical_features, bool):
            self.categorical_features = np.array([self.categorical_features])
        elif self.categorical_features != "auto":
            raise ValueError("categorical_features must be 'auto', array-like or a boolean mask")

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        X, y = self._validate_data(
            X, y, reset=True, validate_separately=False,
            accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite='allow-nan'
        )
        y = check_array(y, ensure_2d=False, dtype=None)
        return X, y

    def _validate_column_types(self, X):
        """Compute the indices of the categorical and continuous features."""
        if self.categorical_features == "auto":
            if not _is_pandas_df(X):
                raise ValueError("When categorical_features is 'auto', X must be a pandas DataFrame")
            self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.continuous_features_ = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        else:
            if isinstance(self.categorical_features, bool):
                self.categorical_features_ = np.arange(X.shape[1])[self.categorical_features]
            else:
                self.categorical_features_ = np.array(self.categorical_features)
            self.continuous_features_ = np.setdiff1d(np.arange(X.shape[1]), self.categorical_features_)
        
        if len(self.categorical_features_) == 0:
            raise ValueError("No categorical features found in the data. SMOTENC cannot be applied.")
        if len(self.continuous_features_) == 0:
            raise ValueError("No continuous features found in the data. Use SMOTEN instead of SMOTENC.")

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps, y_type, y=None):
        """Generate a synthetic sample with an additional steps for the
        categorical features.

        Each new sample is generated the same way than in SMOTE. However, the
        categorical features are mapped to the most frequent nearest neighbors
        of the majority class.
        """
        X_new = np.zeros((steps.shape[0], X.shape[1]))
        
        for feature_idx in range(X.shape[1]):
            if feature_idx in self.continuous_features_:
                X_new[:, feature_idx] = X[rows, feature_idx] + steps * (
                    X[nn_num[rows, cols], feature_idx] - X[rows, feature_idx]
                )
            else:
                X_new[:, feature_idx] = X[nn_num[rows, cols], feature_idx]
        
        return X_new

    @property
    def ohe_(self):
        """One-hot encoder used to encode the categorical features."""
        warnings.warn(
            "`ohe_` is deprecated in 0.11 and will be removed in 0.13. "
            "Use `categorical_encoder_` instead.",
            DeprecationWarning
        )
        return self.categorical_encoder_

@Substitution(sampling_strategy=BaseOverSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring, random_state=_random_state_docstring)
class SMOTEN(SMOTE):
    """Synthetic Minority Over-sampling Technique for Nominal.

    This method is referred as SMOTEN in [1]_. It expects that the data to
    resample are only made of categorical features.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    categorical_encoder : estimator, default=None
        Ordinal encoder used to encode the categorical features. If `None`, a
        :class:`~sklearn.preprocessing.OrdinalEncoder` is used with default parameters.

    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
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
    categorical_encoder_ : estimator
        The encoder used to encode the categorical features.

    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_k_ : estimator object
        Validated k-nearest neighbours created from the `k_neighbors` parameter.

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

    BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

    SVMSMOTE : Over-sample using the SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original papers: [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array(["A"] * 10 + ["B"] * 20 + ["C"] * 30, dtype=object).reshape(-1, 1)
    >>> y = np.array([0] * 20 + [1] * 40, dtype=np.int32)
    >>> from collections import Counter
    >>> print(f"Original class counts: {{Counter(y)}}")
    Original class counts: Counter({{1: 40, 0: 20}})
    >>> from imblearn.over_sampling import SMOTEN
    >>> sampler = SMOTEN(random_state=0)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> print(f"Class counts after resampling {{Counter(y_res)}}")
    Class counts after resampling Counter({{0: 40, 1: 40}})
    """
    _parameter_constraints: dict = {**SMOTE._parameter_constraints, 'categorical_encoder': [HasMethods(['fit_transform', 'inverse_transform']), None]}

    def __init__(self, categorical_encoder=None, *, sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors, n_jobs=n_jobs)
        self.categorical_encoder = categorical_encoder

    def _check_X_y(self, X, y):
        """Check should accept strings and not sparse matrices."""
        if issparse(X):
            raise ValueError("SMOTEN does not support sparse input. Please convert X to a dense numpy array or pandas DataFrame.")
        
        X, y = self._validate_data(
            X, y, reset=True, validate_separately=False,
            dtype=None, force_all_finite=False
        )
        y = check_array(y, ensure_2d=False, dtype=None)
        
        if not np.all([isinstance(x, (str, int, np.integer)) for x in X.ravel()]):
            raise ValueError("SMOTEN supports only string and integer categorical features.")
        
        return X, y

    def _validate_estimator(self):
        """Force to use precomputed distance matrix."""
        self.nn_k_ = check_neighbors_object('k_neighbors', self.k_neighbors, additional_neighbor=1)
        self.nn_k_.set_params(**{'metric': 'precomputed'})
