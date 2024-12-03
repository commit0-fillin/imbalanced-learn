"""Test the module random under sampler."""
from collections import Counter
from datetime import datetime
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import RandomUnderSampler
RND_SEED = 0
X = np.array([[0.04352327, -0.20515826], [0.92923648, 0.76103773], [0.20792588, 1.49407907], [0.47104475, 0.44386323], [0.22950086, 0.33367433], [0.15490546, 0.3130677], [0.09125309, -0.85409574], [0.12372842, 0.6536186], [0.13347175, 0.12167502], [0.094035, -2.55298982]])
Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])

@pytest.mark.parametrize('sampling_strategy', ['auto', 'majority', 'not minority', 'not majority', 'all'])
def test_random_under_sampler_strings(sampling_strategy):
    """Check that we support all supposed strings as `sampling_strategy` in
    a sampler inheriting from `BaseUnderSampler`."""
    X, y = make_classification(
        n_samples=1000, n_classes=3, n_informative=4, weights=[0.2, 0.3, 0.5], random_state=RND_SEED
    )
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=RND_SEED)
    X_res, y_res = rus.fit_resample(X, y)
    
    if sampling_strategy == 'auto' or sampling_strategy == 'not minority':
        assert Counter(y_res)[1] == Counter(y_res)[2]
    elif sampling_strategy == 'majority':
        assert Counter(y_res)[2] == Counter(y_res)[1]
    elif sampling_strategy == 'not majority':
        assert Counter(y_res)[0] == Counter(y_res)[1]
    elif sampling_strategy == 'all':
        assert len(set(Counter(y_res).values())) == 1
    
    assert len(X_res) == len(y_res)

def test_random_under_sampling_datetime():
    """Check that we don't convert input data and only sample from it."""
    X = np.array([datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3),
                  datetime(2022, 1, 4), datetime(2022, 1, 5), datetime(2022, 1, 6)])
    y = np.array([0, 0, 0, 1, 1, 1])

    rus = RandomUnderSampler(sampling_strategy='auto', random_state=RND_SEED)
    X_res, y_res = rus.fit_resample(X.reshape(-1, 1), y)

    assert X_res.dtype == object
    assert all(isinstance(x[0], datetime) for x in X_res)
    assert len(X_res) == 4  # 2 samples from each class
    assert Counter(y_res) == {0: 2, 1: 2}

def test_random_under_sampler_full_nat():
    """Check that we can return timedelta columns full of NaT.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/1055
    """
    X = np.array([np.timedelta64('NaT')] * 6, dtype='timedelta64[ns]').reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])

    rus = RandomUnderSampler(sampling_strategy='auto', random_state=RND_SEED)
    X_res, y_res = rus.fit_resample(X, y)

    assert X_res.dtype == 'timedelta64[ns]'
    assert np.all(np.isnat(X_res))
    assert len(X_res) == 4  # 2 samples from each class
    assert Counter(y_res) == {0: 2, 1: 2}
