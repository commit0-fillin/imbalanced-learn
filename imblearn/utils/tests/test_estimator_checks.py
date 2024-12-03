import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import check_classification_targets
from imblearn.base import BaseSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_target_type as target_check
from imblearn.utils.estimator_checks import check_samplers_fit, check_samplers_nan, check_samplers_one_label, check_samplers_preserve_dtype, check_samplers_sparse, check_samplers_string, check_target_type

class BaseBadSampler(BaseEstimator):
    """Sampler without inputs checking."""
    _sampling_type = 'bypass'

    def fit_resample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

class SamplerSingleClass(BaseSampler):
    """Sampler that would sample even with a single class."""
    _sampling_type = 'bypass'

    def _fit_resample(self, X, y):
        return X, y

class NotFittedSampler(BaseBadSampler):
    """Sampler without target checking."""

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)

    def _fit_resample(self, X, y):
        return X, y

class NoAcceptingSparseSampler(BaseBadSampler):
    """Sampler which does not accept sparse matrix."""

    def fit_resample(self, X, y):
        if sparse.issparse(X):
            raise TypeError("A dense array is required.")
        return X, y

class NotPreservingDtypeSampler(BaseSampler):
    _sampling_type = 'bypass'
    _parameter_constraints: dict = {'sampling_strategy': 'no_validation'}

    def _fit_resample(self, X, y):
        return X.astype(float), y

class IndicesSampler(BaseOverSampler):
    def _fit_resample(self, X, y):
        # This sampler returns the indices of the samples instead of the samples themselves
        indices = np.arange(len(X))
        return indices, y
mapping_estimator_error = {'BaseBadSampler': (AssertionError, 'ValueError not raised by fit'), 'SamplerSingleClass': (AssertionError, "Sampler can't balance when only"), 'NotFittedSampler': (AssertionError, 'No fitted attribute'), 'NoAcceptingSparseSampler': (TypeError, 'dense data is required'), 'NotPreservingDtypeSampler': (AssertionError, 'X dtype is not preserved')}
