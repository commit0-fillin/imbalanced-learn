"""Test the module ."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import InstanceHardnessThreshold
RND_SEED = 0
X = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189], [-0.77740357, 0.74097941], [0.91542919, -0.65453327], [-0.03852113, 0.40910479], [-0.43877303, 1.07366684], [-0.85795321, 0.82980738], [-0.18430329, 0.52328473], [-0.30126957, -0.66268378], [-0.65571327, 0.42412021], [-0.28305528, 0.30284991], [0.20246714, -0.34727125], [1.06446472, -1.09279772], [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
Y = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0])
ESTIMATOR = GradientBoostingClassifier(random_state=RND_SEED)

def test_iht_estimator_pipeline():
    """Check that we can pass a pipeline containing a classifier.

    Checking if we have a classifier should not be based on inheriting from
    `ClassifierMixin`.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/pull/1049
    """
    # Create a pipeline with a classifier
    clf_pipeline = make_pipeline(NB())
    
    # Create an InstanceHardnessThreshold object with the pipeline
    iht = InstanceHardnessThreshold(estimator=clf_pipeline, random_state=RND_SEED)
    
    # Fit and resample
    X_resampled, y_resampled = iht.fit_resample(X, Y)
    
    # Check that resampling occurred (fewer samples than original)
    assert len(X_resampled) < len(X)
    assert len(y_resampled) < len(Y)
    
    # Check that the class distribution has changed
    unique, counts = np.unique(y_resampled, return_counts=True)
    assert len(unique) == len(np.unique(Y))  # All classes are still present
    assert not np.all(counts == np.unique(Y, return_counts=True)[1])  # Counts have changed
