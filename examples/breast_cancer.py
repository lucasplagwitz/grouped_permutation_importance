from sklearn.datasets import load_breast_cancer
from grouped_permutation_importance.grouped_permutation_importance import grouped_permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

data = load_breast_cancer()
feature_names = data["feature_names"].tolist()
X, y = data["data"], data["target"]

idxs = []
columns = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity",  "concave", "symmetry", "fractal"]
for key in columns:
    idxs.append([x for (x, y) in enumerate(feature_names) if key in y])

model = Pipeline([("MinMax", MinMaxScaler()),  ("SVC", SVC())])
cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2)

r = grouped_permutation_importance(model, X, y, idxs=idxs,
                                   n_repeats=50, random_state=0, scoring="balanced_accuracy", n_jobs=5,
                                   cv=cv, perm_set="train")

sorted_idx = r.importances_mean.argsort()
fig, ax = plt.subplots(2, 2)

ax[0, 0].boxplot(r.importances[sorted_idx].T,
           vert=False, labels=np.array(columns)[sorted_idx])
ax[0, 0].set_title("Importances (train set)")


r = grouped_permutation_importance(model, X, y, idxs=idxs,
                              n_repeats=50, random_state=0, scoring="balanced_accuracy", n_jobs=5,
                                  cv=cv, perm_set="test")


ax[0, 1].boxplot(r.importances[sorted_idx].T,
           vert=False, labels=np.array(columns)[sorted_idx])
ax[0, 1].set_title("Importances (test set)")

columns = ["mean", "error", "worst"]
idxs = []
for key in columns:
    idxs.append([x for (x, y) in enumerate(feature_names) if key in y])

r = grouped_permutation_importance(model, X, y, idxs=idxs,
                              n_repeats=50, random_state=0, scoring="balanced_accuracy", n_jobs=5,
                                  cv=cv, perm_set="train")
sorted_idx = r.importances_mean.argsort()
ax[1, 0].boxplot(r.importances[sorted_idx].T,
           vert=False, labels=np.array(columns)[sorted_idx])
ax[1, 0].set_title("Importances (train set)")


r = grouped_permutation_importance(model, X, y, idxs=idxs,
                              n_repeats=50, random_state=0, scoring="balanced_accuracy", n_jobs=5,
                                  cv=cv, perm_set="test")


ax[1, 1].boxplot(r.importances[sorted_idx].T,
           vert=False, labels=np.array(columns)[sorted_idx])
ax[1, 1].set_title("Importances (test set)")
fig.tight_layout()
plt.savefig("../demo/breast_cancer.png")
