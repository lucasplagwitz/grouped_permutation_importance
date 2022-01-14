from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

from grouped_permutation_importance import grouped_permutation_importance

params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'sans-serif',
          'figure.figsize': (17,7)
          }
plt.rcParams.update(params)

data = load_breast_cancer()
feature_names = data["feature_names"].tolist()
X, y = data["data"], data["target"]

model = Pipeline([("MinMax", MinMaxScaler()),  ("SVC", SVC())])
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})

# first grouping
columns = ["mean", "error", "worst"]
idxs = []
for key in columns:
    idxs.append([x for (x, y) in enumerate(feature_names) if key in y])

r = grouped_permutation_importance(model, X, y, idxs=idxs,
                                   n_repeats=100, random_state=0,
                                   scoring="balanced_accuracy", n_jobs=5,
                                   cv=cv, perm_set="test")
sorted_idx = r.importances_mean.argsort()[::-1]
box = ax[0].boxplot(r.importances[sorted_idx].T,
                    patch_artist=True,
                    vert=True,
                    labels=np.array(columns)[sorted_idx])
for patch in box["boxes"]:
    patch.set_facecolor("blue")
    patch.set_alpha(.5)
ax[0].set_ylabel("information gain")

# different grouping
columns = ["radius", "texture", "perimeter", "area", "smoothness",
           "compactness", "concavity",  "concave", "symmetry", "fractal"]
idxs = []
for key in columns:
    idxs.append([x for (x, y) in enumerate(feature_names) if key in y])
r = grouped_permutation_importance(model, X, y, idxs=idxs,
                                   n_repeats=50, random_state=0,
                                   scoring="balanced_accuracy", n_jobs=5,
                                   cv=cv, perm_set="test")
sorted_idx = r.importances_mean.argsort()[::-1]
box = ax[1].boxplot(r.importances[sorted_idx].T,
                    patch_artist=True,
                    vert=True, labels=np.array(columns)[sorted_idx])
for patch in box['boxes']:
    patch.set_facecolor("blue")
    patch.set_alpha(.5)

ax[1].set_xlabel("feature subset")
ax[1].xaxis.set_label_coords(0.2, -0.125)

fig.tight_layout()
plt.savefig("../demo/breast_cancer.png")
