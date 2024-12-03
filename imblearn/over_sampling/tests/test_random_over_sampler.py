"""Test the module under sampler."""
from collections import Counter
from datetime import datetime
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import _convert_container, assert_allclose, assert_array_equal
from imblearn.over_sampling import RandomOverSampler
RND_SEED = 0

@pytest.mark.parametrize('sampling_strategy', ['auto', 'minority', 'not minority', 'not majority', 'all'])
def test_random_over_sampler_strings(sampling_strategy):
    """Check that we support all supposed strings as `sampling_strategy` in
    a sampler inheriting from `BaseOverSampler`."""
    X, y = make_classification(
        n_samples=100,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=RND_SEED,
    )

    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RND_SEED)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    assert isinstance(X_resampled, np.ndarray)
    assert isinstance(y_resampled, np.ndarray)
    
    if sampling_strategy == 'auto' or sampling_strategy == 'all':
        assert len(np.unique(y_resampled)) == len(np.unique(y))
        assert all(np.bincount(y_resampled) == np.max(np.bincount(y_resampled)))
    elif sampling_strategy == 'minority':
        assert np.sum(y_resampled == np.argmin(np.bincount(y))) == np.max(np.bincount(y))
    elif sampling_strategy == 'not minority':
        assert np.all(np.bincount(y_resampled)[1:] == np.max(np.bincount(y)))
    elif sampling_strategy == 'not majority':
        assert np.all(np.bincount(y_resampled)[:-1] == np.max(np.bincount(y)))

def test_random_over_sampling_datetime():
    """Check that we don't convert input data and only sample from it."""
    X = np.array([datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3),
                  datetime(2022, 1, 4), datetime(2022, 1, 5)]).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1])

    ros = RandomOverSampler(random_state=RND_SEED)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    assert X_resampled.dtype == X.dtype
    assert len(X_resampled) == 6
    assert len(y_resampled) == 6
    assert np.sum(y_resampled == 0) == 3
    assert np.sum(y_resampled == 1) == 3
    assert all(x in X.flatten() for x in X_resampled.flatten())

def test_random_over_sampler_full_nat():
    """Check that we can return timedelta columns full of NaT.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/1055
    """
    X = np.array([np.timedelta64('NaT'), np.timedelta64('NaT')]).reshape(-1, 1)
    y = np.array([0, 1])

    ros = RandomOverSampler(random_state=RND_SEED)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    assert X_resampled.dtype == X.dtype
    assert len(X_resampled) == 2
    assert len(y_resampled) == 2
    assert np.sum(y_resampled == 0) == 1
    assert np.sum(y_resampled == 1) == 1
    assert np.all(np.isnat(X_resampled))
