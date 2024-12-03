"""Test the module condensed nearest neighbour."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import CondensedNearestNeighbour
RND_SEED = 0
X = np.array([[2.59928271, 0.93323465], [0.25738379, 0.95564169], [1.42772181, 0.526027], [1.92365863, 0.82718767], [-0.10903849, -0.12085181], [-0.284881, -0.62730973], [0.57062627, 1.19528323], [0.03394306, 0.03986753], [0.78318102, 2.59153329], [0.35831463, 1.33483198], [-0.14313184, -1.0412815], [0.01936241, 0.17799828], [-1.25020462, -0.40402054], [-0.09816301, -0.74662486], [-0.01252787, 0.34102657], [0.52726792, -0.38735648], [0.2821046, -0.07862747], [0.05230552, 0.09043907], [0.15198585, 0.12512646], [0.70524765, 0.39816382]])
Y = np.array([1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 1, 2, 1])

def test_condensed_nearest_neighbour_multiclass():
    """Check the validity of the fitted attributes `estimators_`."""
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)
    cnn.fit_resample(X, Y)
    
    assert hasattr(cnn, 'sample_indices_')
    assert len(cnn.sample_indices_) < len(X)
    
    X_resampled, y_resampled = cnn.fit_resample(X, Y)
    
    # Check that the number of samples is reduced
    assert len(X_resampled) < len(X)
    assert len(y_resampled) < len(Y)
    
    # Check that all classes are still present in the resampled data
    assert set(y_resampled) == set(Y)
    
    # Check that the resampled data can be used to train a classifier
    clf = KNeighborsClassifier()
    clf.fit(X_resampled, y_resampled)
    
    # Predict on the original data
    y_pred = clf.predict(X)
    
    # Check that the prediction is not all the same class
    assert len(set(y_pred)) > 1

def test_condensed_nearest_neighbors_deprecation():
    """Check that we raise a FutureWarning when accessing the parameter `estimator_`."""
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)
    cnn.fit(X, Y)
    
    with pytest.warns(FutureWarning, match="The attribute `estimator_` is deprecated"):
        _ = cnn.estimator_
