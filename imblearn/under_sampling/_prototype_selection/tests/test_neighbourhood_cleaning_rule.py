"""Test the module neighbourhood cleaning rule."""
from collections import Counter
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import EditedNearestNeighbours, NeighbourhoodCleaningRule

def test_ncr_threshold_cleaning(data):
    """Test the effect of the `threshold_cleaning` parameter."""
    X, y = data
    
    # Test with default threshold_cleaning (0.5)
    ncr = NeighbourhoodCleaningRule(random_state=42)
    X_resampled, y_resampled = ncr.fit_resample(X, y)
    
    # Test with a higher threshold_cleaning (0.8)
    ncr_high = NeighbourhoodCleaningRule(threshold_cleaning=0.8, random_state=42)
    X_resampled_high, y_resampled_high = ncr_high.fit_resample(X, y)
    
    # Test with a lower threshold_cleaning (0.2)
    ncr_low = NeighbourhoodCleaningRule(threshold_cleaning=0.2, random_state=42)
    X_resampled_low, y_resampled_low = ncr_low.fit_resample(X, y)
    
    # Check that higher threshold_cleaning removes fewer samples
    assert len(y_resampled_high) > len(y_resampled)
    
    # Check that lower threshold_cleaning removes more samples
    assert len(y_resampled_low) < len(y_resampled)
    
    # Check that the class distribution is affected by the threshold
    assert Counter(y_resampled_high) != Counter(y_resampled_low)

def test_ncr_n_neighbors(data):
    """Check the effect of the NN on the cleaning of the second phase."""
    X, y = data
    
    # Test with default n_neighbors (None, which uses 3)
    ncr_default = NeighbourhoodCleaningRule(random_state=42)
    X_resampled_default, y_resampled_default = ncr_default.fit_resample(X, y)
    
    # Test with a higher number of neighbors
    ncr_high = NeighbourhoodCleaningRule(n_neighbors=5, random_state=42)
    X_resampled_high, y_resampled_high = ncr_high.fit_resample(X, y)
    
    # Test with a lower number of neighbors
    ncr_low = NeighbourhoodCleaningRule(n_neighbors=1, random_state=42)
    X_resampled_low, y_resampled_low = ncr_low.fit_resample(X, y)
    
    # Check that different n_neighbors values produce different results
    assert not np.array_equal(X_resampled_default, X_resampled_high)
    assert not np.array_equal(X_resampled_default, X_resampled_low)
    assert not np.array_equal(X_resampled_high, X_resampled_low)
    
    # Check that the class distribution is affected by n_neighbors
    assert Counter(y_resampled_default) != Counter(y_resampled_high)
    assert Counter(y_resampled_default) != Counter(y_resampled_low)
    
    # Check that the number of samples removed is different for each case
    assert len(y_resampled_default) != len(y_resampled_high)
    assert len(y_resampled_default) != len(y_resampled_low)
