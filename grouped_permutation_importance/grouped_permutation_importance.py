# The code is 99.9 percent from scikit learn.
import numpy as np
from joblib import Parallel

from sklearn.metrics import check_scoring
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils.fixes import delayed
from sklearn.inspection._permutation_importance import _weights_scorer
from grouped_permutation_importance._adapted_permutation_importance import _calculate_permutation_scores

def grouped_permutation_importance(estimator, X, y, *, scoring=None, n_repeats=5, idxs=None,
                                   n_jobs=None, random_state=None, sample_weight=None):

    if not hasattr(X, "iloc"):
        X = check_array(X, force_all_finite='allow-nan', dtype=None)

    # Precompute random seed from the random state to be used
    # to get a fresh independent RandomState instance for each
    # parallel call to _calculate_permutation_scores, irrespective of
    # the fact that variables are shared or not depending on the active
    # joblib backend (sequential, thread-based or process-based).
    random_state = check_random_state(random_state)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    scorer = check_scoring(estimator, scoring=scoring)
    baseline_score = _weights_scorer(scorer, estimator, X, y, sample_weight)

    scores = Parallel(n_jobs=n_jobs)(delayed(_calculate_permutation_scores)(
        estimator, X, y, sample_weight, col_idx, random_seed, n_repeats, scorer
    ) for col_idx in idxs)

    importances = baseline_score - np.array(scores)
    return Bunch(importances_mean=np.mean(importances, axis=1),
                 importances_std=np.std(importances, axis=1),
                 importances=importances)