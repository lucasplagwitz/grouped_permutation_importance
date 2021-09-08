# Grouped Permutation Importance

The interpretability of machine learning models is a common task. 
In many domains datasets consist of different feature sources.  
This repo allows a simple analysis to calculate the influence 
of a feature group on the overall result. This is done by a slight 
modification of the permutation importance of scikit-learn. 

```python
data = load_breast_cancer()
feature_names = data["feature_names"].tolist()
X, y = data["data"], data["target"]

idxs = []
columns = ["mean", "error", "worst"]
for key in columns:
    idxs.append([x for (x, y) in enumerate(feature_names) if key in y])

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=.2)

pipe = Pipeline([("MinMax", MinMaxScaler()),  ("SVC", SVC())])
model = pipe.fit(X_train, y_train)


r = grouped_permutation_importance(model, X_train, y_train, idxs=idxs,
                                   n_repeats=50, random_state=0, scoring="balanced_accuracy", n_jobs=5)
```

<p align="center">
<img src="./demo/breast_cancer.png">
</p>