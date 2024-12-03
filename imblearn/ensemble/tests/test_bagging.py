"""Test the module ensemble classifiers."""
from collections import Counter
import numpy as np
import pytest
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_classification, make_hastie_10_2
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.utils.fixes import parse_version
from imblearn import FunctionSampler
from imblearn.datasets import make_imbalance
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
sklearn_version = parse_version(sklearn.__version__)
iris = load_iris()

class CountDecisionTreeClassifier(DecisionTreeClassifier):
    """DecisionTreeClassifier that will memorize the number of samples seen
    at fit."""

def test_balanced_bagging_classifier_n_features():
    """Check that we raise a FutureWarning when accessing `n_features_`."""
    X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, random_state=0)
    bbc = BalancedBaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
    bbc.fit(X, y)
    
    with pytest.warns(FutureWarning, match="The `n_features_` attribute is deprecated"):
        _ = bbc.n_features_
