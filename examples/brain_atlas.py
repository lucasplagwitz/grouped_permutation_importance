import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (25,7)

from nilearn import datasets
from nilearn.regions import RegionExtractor

from nilearn.datasets import fetch_oasis_vbm



from grouped_permutation_importance import grouped_permutation_importance

smith_atlas = datasets.fetch_atlas_smith_2009()
atlas_networks = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm').maps
labels = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm').labels

print(labels)
# GET DATA FROM OASIS
n_subjects = 403
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
sex = np.array(dataset_files.ext_vars["mf"]).astype(str)
age = np.array(dataset_files.ext_vars["age"]).astype(float)
cdr = dataset_files.ext_vars["cdr"].astype(float)
y_cdr = (np.array(cdr)==0.5).astype(int)*1 + (np.array(cdr)==1).astype(int)*1
y_sex = (sex == "F").astype(int)
y_age = (age > 50).astype(int)

save_file = "brain_regions.npz"
if not os.path.isfile(save_file):
    # min_region_size in voxel volume mm^3
    extraction = RegionExtractor(atlas_networks,  #min_region_size=200 ,  extractor='local_regions',
                                 threshold=.5, #thresholding_strategy='percentile',
                                 standardize=True)

    # Just call fit() to execute region extraction procedure
    extraction.fit()
    regions_img = extraction.regions_img_



    X = np.array(dataset_files.gray_matter_maps)

    Z = extraction.transform(X)
    e_index = extraction.index_
    np.savez_compressed(save_file, Z=Z, y_age=y_age, y_cdr=y_cdr, y_sex = y_sex, e_index=e_index)
else:
    data = np.load(save_file)
    Z = data["Z"]
    y_age = data["y_age"]
    y_sex = data["y_sex"]
    y_cdr = data["y_cdr"]
    e_index = data["e_index"]

cv=StratifiedShuffleSplit(n_splits=45, test_size=0.2)
for plot_i, y in enumerate([y_age, y_cdr, y_sex]):

    model = Pipeline([("Scaler", StandardScaler()),
                      ("SVC", SVC(class_weight="balanced"))])

    idxs = []
    for key in ["Cerebral Cortex ", "Cerebral White Matter", "Thalamus", "Amygdala", "Hippocampus", "Pallidum", "Lateral Ventrical", "Putamen", "Brain-Stem", "Background"]:
        t = [l for l, val in enumerate(labels) if key in val]
        idxs.append(np.nonzero(sum([np.array(e_index) == i for i in t]))[0].tolist())

    columns = ["Cerebral Cortex ", "Cerebral White Matter", "Thalamus", "Amygdala", "Hippocampus", "Pallidum", "Lateral Ventrical", "Putamen", "Brain-Stem", "Background"]

    r = grouped_permutation_importance(model, Z, y, idxs=idxs,
                                       n_repeats=100, random_state=0, scoring="balanced_accuracy", n_jobs=5,
                                       cv=cv, perm_set="test", verbose=1, min_performance=0.75)

    if plot_i == 0:
        sorted_idx = r.importances_mean.argsort()

    plt.subplot(1, 3, plot_i+1)

    plt.boxplot(r.importances[sorted_idx].T,
               vert=False, labels=np.array(columns)[sorted_idx])
    plt.vlines(x=0, ymin=1-0.3, ymax=len(columns)+0.3, linestyles="dashed", colors="black")

    if plot_i == 0:
        plt.title("Age-Prediction - Importances (test set)")
    elif plot_i == 1:
        plt.title("Alzheimer-Prediction - Importances (test set)")
    else:
        plt.title("Sex-Prediction - Importances (test set)")

    plt.tight_layout(pad=3.5)

plt.savefig("../demo/brain_atlas.png")