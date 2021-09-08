from sklearn.datasets import load_breast_cancer
from grouped_permutation_importance.grouped_permutation_importance import grouped_permutation_importance
from sklearn.model_selection import train_test_split
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

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=.2)

pipe = Pipeline([("MinMax", MinMaxScaler()),  ("SVC", SVC())])
model = pipe.fit(X_train, y_train)
print(pipe.score(X_val, y_val))


r = grouped_permutation_importance(model, X_train, y_train, idxs=idxs,
                                   n_repeats=50, random_state=0, scoring="balanced_accuracy", n_jobs=5)

sorted_idx = r.importances_mean.argsort()
fig, ax = plt.subplots(2, 2)

ax[0, 0].boxplot(r.importances[sorted_idx].T,
           vert=False, labels=np.array(columns)[sorted_idx])
ax[0, 0].set_title("Importances (train set)")


r = grouped_permutation_importance(model, X_val, y_val, idxs=idxs,
                              n_repeats=50, random_state=0, scoring="balanced_accuracy", n_jobs=5)


ax[0, 1].boxplot(r.importances[sorted_idx].T,
           vert=False, labels=np.array(columns)[sorted_idx])
ax[0, 1].set_title("Importances (test set)")

columns = ["mean", "error", "worst"]
idxs = []
for key in columns:
    idxs.append([x for (x, y) in enumerate(feature_names) if key in y])

r = grouped_permutation_importance(model, X_train, y_train, idxs=idxs,
                              n_repeats=50, random_state=0, scoring="balanced_accuracy", n_jobs=5)
sorted_idx = r.importances_mean.argsort()
ax[1, 0].boxplot(r.importances[sorted_idx].T,
           vert=False, labels=np.array(columns)[sorted_idx])
ax[1, 0].set_title("Importances (train set)")


r = grouped_permutation_importance(model, X_val, y_val, idxs=idxs,
                              n_repeats=50, random_state=0, scoring="balanced_accuracy", n_jobs=5)


ax[1, 1].boxplot(r.importances[sorted_idx].T,
           vert=False, labels=np.array(columns)[sorted_idx])
ax[1, 1].set_title("Importances (test set)")
fig.tight_layout()
plt.savefig("../demo/breast_cancer.png")
