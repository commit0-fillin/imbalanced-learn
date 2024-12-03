import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.utils._testing import assert_allclose, assert_array_equal
from imblearn.over_sampling import SVMSMOTE

def test_svm_smote_not_svm(data):
    """Check that we raise a proper error if passing an estimator that does not
    expose a `support_` fitted attribute."""
    X, y = data
    svm_smote = SVMSMOTE(svm_estimator=LogisticRegression())
    with pytest.raises(ValueError, match="The svm_estimator doesn't expose a"):
        svm_smote.fit_resample(X, y)

def test_svm_smote_all_noise(data):
    """Check that we raise a proper error message when all support vectors are
    detected as noise and there is nothing that we can do.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/742
    """
    X, y = data
    # Create a situation where all support vectors are detected as noise
    svm = SVC(kernel="rbf", gamma=1e-3)
    svm_smote = SVMSMOTE(svm_estimator=svm, n_neighbors=1)
    
    with pytest.raises(RuntimeError, match="No support vectors were found"):
        svm_smote.fit_resample(X, y)
