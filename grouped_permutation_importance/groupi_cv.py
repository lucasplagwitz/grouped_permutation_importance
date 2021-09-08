from sklearn.metrics import check_scoring
from sklearn.inspection._permutation_importance import _weights_scorer

class GroupiCV(object):
    """
    Cross Validation for Grouped Permutation Importance.
    """

    def __init__(self, estimator, scoring=None, n_repeats=5, idxs=None, cv=None):
        raise NotImplementedError("Coming soon!")
        self.cv_results_ = None
        self.estimator = estimator
        self.scorer = check_scoring(estimator, scoring=scoring)


    def fit(self, X, y):
        pass