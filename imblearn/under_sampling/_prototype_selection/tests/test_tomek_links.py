"""Test the module Tomek's links."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import TomekLinks
X = np.array([[0.31230513, 0.1216318], [0.68481731, 0.51935141], [1.34192108, -0.13367336], [0.62366841, -0.21312976], [1.61091956, -0.40283504], [-0.37162401, -2.19400981], [0.74680821, 1.63827342], [0.2184254, 0.24299982], [0.61472253, -0.82309052], [0.19893132, -0.47761769], [1.06514042, -0.0770537], [0.97407872, 0.44454207], [1.40301027, -0.83648734], [-1.20515198, -1.02689695], [-0.27410027, -0.54194484], [0.8381014, 0.44085498], [-0.23374509, 0.18370049], [-0.32635887, -0.29299653], [-0.00288378, 0.84259929], [1.79580611, -0.02219234]])
Y = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])

@pytest.mark.parametrize('sampling_strategy', ['auto', 'majority', 'not minority', 'not majority', 'all'])
def test_tomek_links_strings(sampling_strategy):
    """Check that we support all supposed strings as `sampling_strategy` in
    a sampler inheriting from `BaseCleaningSampler`."""
    tomek = TomekLinks(sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = tomek.fit_resample(X, Y)
    
    # Check that the resampled data is different from the original
    assert not np.array_equal(X_resampled, X) or not np.array_equal(y_resampled, Y)
    
    # Check that the resampled data has the same number of features
    assert X_resampled.shape[1] == X.shape[1]
    
    # Check that the number of samples in X_resampled matches y_resampled
    assert X_resampled.shape[0] == y_resampled.shape[0]
    
    # Check that the resampled data contains only the original class labels
    assert set(np.unique(y_resampled)).issubset(set(np.unique(Y)))
