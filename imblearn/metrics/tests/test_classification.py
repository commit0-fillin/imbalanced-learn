"""Testing the metric for classification with imbalanced dataset"""
from functools import partial
import numpy as np
import pytest
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, cohen_kappa_score, jaccard_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils._testing import assert_allclose, assert_array_equal, assert_no_warnings
from sklearn.utils.validation import check_random_state
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score, macro_averaged_mean_absolute_error, make_index_balanced_accuracy, sensitivity_score, sensitivity_specificity_support, specificity_score
RND_SEED = 42
R_TOL = 0.01

def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC
    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """
    if dataset is None:
        # Create a toy dataset
        X, y = datasets.make_classification(
            n_classes=2 if binary else 3,
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_repeated=0,
            n_clusters_per_class=2,
            weights=[0.7, 0.3] if binary else [0.5, 0.3, 0.2],
            random_state=RND_SEED,
        )
    else:
        X, y = dataset

    # Split the data into training and testing sets
    random_state = check_random_state(RND_SEED)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    random_state.shuffle(indices)
    X = X[indices]
    y = y[indices]
    half = int(n_samples / 2)

    # Fit an SVC on the data
    clf = svm.SVC(kernel='linear', probability=True, random_state=RND_SEED)
    clf.fit(X[:half], y[:half])

    # Predict on the remaining data
    y_pred = clf.predict(X[half:])
    y_true = y[half:]

    return y_true, y_pred, clf.decision_function(X[half:])
