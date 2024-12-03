from collections import Counter
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import assert_allclose, assert_array_equal
from imblearn.over_sampling import BorderlineSMOTE

@pytest.mark.parametrize('kind', ['borderline-1', 'borderline-2'])
def test_borderline_smote_no_in_danger_samples(kind):
    """Check that the algorithm behave properly even on a dataset without any sample
    in danger.
    """
    X, y = make_classification(
        n_samples=100,
        n_classes=2,
        weights=[0.9, 0.1],
        n_informative=3,
        n_redundant=1,
        n_repeated=0,
        n_features=20,
        n_clusters_per_class=1,
        random_state=0,
    )

    sm = BorderlineSMOTE(kind=kind, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    assert Counter(y_res)[0] == Counter(y_res)[1]
    assert X_res.shape[0] == y_res.shape[0]
    assert X_res.shape[1] == X.shape[1]

def test_borderline_smote_kind():
    """Check the behaviour of the `kind` parameter.

    In short, "borderline-2" generates sample closer to the boundary decision than
    "borderline-1". We generate an example where a logistic regression will perform
    worse on "borderline-2" than on "borderline-1".
    """
    X, y = make_classification(
        n_samples=100,
        n_classes=2,
        weights=[0.9, 0.1],
        n_informative=3,
        n_redundant=1,
        n_repeated=0,
        n_features=20,
        n_clusters_per_class=1,
        random_state=0,
    )

    sm1 = BorderlineSMOTE(kind='borderline-1', random_state=42)
    X_res1, y_res1 = sm1.fit_resample(X, y)

    sm2 = BorderlineSMOTE(kind='borderline-2', random_state=42)
    X_res2, y_res2 = sm2.fit_resample(X, y)

    lr1 = LogisticRegression(random_state=42)
    lr1.fit(X_res1, y_res1)
    score1 = lr1.score(X, y)

    lr2 = LogisticRegression(random_state=42)
    lr2.fit(X_res2, y_res2)
    score2 = lr2.score(X, y)

    assert score1 > score2, "borderline-1 should perform better than borderline-2"
