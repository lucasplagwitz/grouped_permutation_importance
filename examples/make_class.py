import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from grouped_permutation_importance import grouped_permutation_importance


X, y = make_classification(n_samples=300, n_features=10, n_informative=10, n_redundant=0, random_state=42)
X = np.concatenate([X, np.random.normal(size=(len(X), 90))], axis=1)

for i, split in enumerate([2, 8, 10]):

    first = range(split)
    second = range(split, 100)
    columns = [f"{split*10}% informative", f"{(10-split)*10}% informative"]


    cv = StratifiedShuffleSplit(5, test_size=0.2)
    test_forest_imp = np.empty((len(columns), 0))
    bacc = []

    model = RandomForestClassifier(class_weight="balanced")

    r_train = grouped_permutation_importance(model, X, y, idxs=[first, second],
                                             n_repeats=30, random_state=0, scoring="balanced_accuracy",
                                             n_jobs=5, cv=cv, perm_set="train")["importances"]


    r_test = grouped_permutation_importance(model, X, y, idxs=[first, second],
                                            n_repeats=30, random_state=0, scoring="balanced_accuracy",
                                            n_jobs=5, cv=cv, perm_set="test")["importances"]


    sorted_idx = list(range(len(columns)))[::-1]

    plt.subplot(3, 2, 1+2*i)
    plt.boxplot(r_train[sorted_idx].T,
               vert=False, labels=np.array(columns)[sorted_idx])
    if i == 0:
        plt.title("Perm-Imp (train set)")

    plt.subplot(3, 2, 2 + 2*i)
    sorted_idx = list(range(len(columns)))[::-1]
    plt.boxplot(r_test[sorted_idx].T,
               vert=False, labels=np.array(columns)[sorted_idx])
    if i == 0:
        plt.title("Perm-Imp (test set)")
plt.tight_layout()
plt.savefig("../demo/make_class.png")