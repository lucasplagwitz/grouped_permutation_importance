import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.datasets import make_classification

from grouped_permutation_importance import grouped_permutation_importance

params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'sans-serif',
          'figure.figsize': (17,7)
          }
plt.rcParams.update(params)

X, y = make_classification(n_samples=300, n_features=10, n_informative=10,
                           n_redundant=0, random_state=42)
X = np.concatenate([X, np.random.normal(size=(len(X), 90))], axis=1)

fig = plt.Figure()

for i, split in enumerate([2, 8, 10]):
    print(f"{i+1} -- {3}")

    first = range(split)
    second = range(split, 100)
    columns = [f"{split*10}\% \n ({len(first)})",
               f"{(10-split)*10}\% \n ({len(second)})"]

    cv = StratifiedShuffleSplit(25, test_size=0.2)

    model = SVC(class_weight="balanced")

    r_train = grouped_permutation_importance(model, X, y, idxs=[first, second],
                                             n_repeats=100, random_state=0,
                                             scoring="balanced_accuracy",
                                             n_jobs=5, cv=cv, perm_set="train",
                                             mode="rel")["importances"]

    r_test = grouped_permutation_importance(model, X, y, idxs=[first, second],
                                            n_repeats=100, random_state=0,
                                            scoring="balanced_accuracy",
                                            n_jobs=5, cv=cv, perm_set="test",
                                            mode="rel")["importances"]

    # -- only plotting
    plt.subplot(1, 3, i+1)
    sorted_idx = list(range(len(columns)))
    bp1 = plt.boxplot(r_train[sorted_idx].T, patch_artist=True,
                      vert=True, labels=["", ""], positions=[2, 5],
                      showfliers=False, notch=True)
    for patch, color in zip(bp1['boxes'], ["blue"]*2):
        patch.set_facecolor(color)
        patch.set_alpha(.6)
    bp2 = plt.boxplot(r_test[sorted_idx].T, patch_artist=True,
                      vert=True, labels=["", ""], positions=[3, 6],
                      showfliers=False, notch=True)
    for patch, color in zip(bp2['boxes'], ["green"]*2):
        patch.set_facecolor(color)
        patch.set_alpha(.7)
    plt.xticks([2.5, 5.5], columns)
    plt.xlim([1, 7])
    plt.ylim([-0.3, 1.6])
    plt.hlines(y=0, xmin=1, xmax=7, linestyles="dashed", colors="red", alpha=.5)
    plt.hlines(y=1, xmin=1, xmax=7, linestyles="dashed", colors="red", alpha=.5)
    if i > 0:
        plt.yticks([])
    else:
        plt.ylabel("relative information gain")
    if i == 2:
        plt.legend([bp1["boxes"][0], bp2["boxes"][0]],
                   [r'train set', r'test set'])
    if i == 1:
        plt.xlabel("of all informative columns \n (number of columns)",
                   labelpad=20)

plt.tight_layout()
plt.savefig("../demo/make_class.png")
