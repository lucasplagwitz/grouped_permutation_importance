import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt

# please install nilearn
from nilearn import datasets
from nilearn.regions import RegionExtractor
from nilearn.datasets import fetch_oasis_vbm

from grouped_permutation_importance import grouped_permutation_importance

params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'sans-serif',
          'figure.figsize': (20,7)
          }
plt.rcParams.update(params)

atlas_networks = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm').maps
labels = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm').labels

# GET DATA FROM OASIS
n_subjects = 403
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
sex = np.array(dataset_files.ext_vars["mf"]).astype(str)
age = np.array(dataset_files.ext_vars["age"]).astype(float)
cdr = dataset_files.ext_vars["cdr"].astype(float)
y_cdr = (np.array(cdr) >= 0.5).astype(int)
y_sex = (sex == "F").astype(int)
y_age = (age > 50).astype(int)

save_file = "brain_regions2.npz"
if not os.path.isfile(save_file):
    # min_region_size in voxel volume mm^3
    extraction = RegionExtractor(atlas_networks, threshold=.5, standardize=True)
    extraction.fit()
    regions_img = extraction.regions_img_
    X = np.array(dataset_files.gray_matter_maps)
    Z = extraction.transform(X)
    e_index = extraction.index_
    print(regions_img)
    print(e_index)
    np.savez_compressed(save_file, Z=Z, y_age=y_age, y_cdr=y_cdr,
                        y_sex=y_sex, e_index=e_index)
else:
    data = np.load(save_file)
    Z = data["Z"]
    y_age = data["y_age"]
    y_sex = data["y_sex"]
    y_cdr = data["y_cdr"]
    e_index = data["e_index"]

cv = RepeatedStratifiedKFold()
imp = []
for plot_i, y in enumerate([y_age, y_cdr, y_sex]):

    model = Pipeline([("Scaler", StandardScaler()),
                      ("SVC", SVC(kernel="rbf", class_weight="balanced"))])

    idxs = []
    columns = ["Cerebral Cortex ", "Cerebral White Matter", "Thalamus",
               "Amygdala", "Hippocampus", "Pallidum",
               "Lateral Ventrical", "Putamen", "Brain-Stem", "Background"]
    for key in columns:
        t = [l for l, val in enumerate(labels) if key in val]
        idxs.append(np.nonzero(sum([np.array(e_index) == i
                                    for i in t]))[0].tolist())

    imp.append(grouped_permutation_importance(model, Z, y, idxs=idxs,
                                              n_repeats=100, random_state=0,
                                              scoring="balanced_accuracy",
                                              n_jobs=8, cv=cv, perm_set="test",
                                              verbose=1, min_performance=0.7,
                                              mode="rel"))

sorted_idx = imp[0].importances_mean.argsort()[::-1]

positions = np.array([1, 5, 9, 13, 17, 21, 25, 29, 33, 37])
bps = []
for j in range(3):
    bps.append(plt.boxplot(imp[j].importances[sorted_idx].T, labels=[""] * 10,
                           showfliers=False, notch=True,
                           patch_artist=True, vert=True,
                           positions=positions + j))

for j, color in enumerate(["green", "blue", "red"]):
    for patch in bps[j]['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(.7)

plt.xticks(positions+1, np.array(columns)[sorted_idx])
plt.legend([x["boxes"][0] for x in bps], ["Age Prediction",
                                          "Alzheimer's Prediction",
                                          "Sex Prediction"])
plt.hlines(y=0, xmin=0.5, xmax=len(columns)*4-.5,
           linestyles="dashed", colors="black")
plt.ylabel("relative information gain")

plt.tight_layout()
plt.savefig("../demo/brain_atlas.png")
