# The code is 99.9 percent from scikit learn.
import numpy as np
from sklearn.inspection._permutation_importance import check_random_state, _weights_scorer


def _calculate_permutation_scores(estimator, X, y, sample_weight, col_idx,
                                  random_state, n_repeats, scorer):
    """Calculate score when `col_idx` is permuted."""
    random_state = check_random_state(random_state)

    # Work on a copy of X to to ensure thread-safety in case of threading based
    # parallelism. Furthermore, making a copy is also useful when the joblib
    # backend is 'loky' (default) or the old 'multiprocessing': in those cases,
    # if X is large it will be automatically be backed by a readonly memory map
    # (memmap). X.copy() on the other hand is always guaranteed to return a
    # writable data-structure whose columns can be shuffled inplace.
    X_permuted = X.copy()
    scores = np.zeros(n_repeats)
    shuffling_idx = np.arange(X.shape[0])
    for n_round in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, "iloc"):
            raise NotImplementedError("DataFrames not yet implemented.")
        else:
            X_permuted[:, col_idx] = X_permuted[[[x] for x in shuffling_idx], col_idx]  # shuffle every column with own rs/seed?
        feature_score = _weights_scorer(
            scorer, estimator, X_permuted, y, sample_weight
        )
        scores[n_round] = feature_score

    return scores
