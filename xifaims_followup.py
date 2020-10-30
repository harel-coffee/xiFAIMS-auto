# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:56:23 2020

@author: hanjo
"""
import os
import pandas as pd
import yaml
import sys

import seaborn as sns
import numpy as np
from xifaims import processing as xp
from scipy.stats import pearsonr
from xifaims import features as xf
from xifaims import plots as xpl
from xifaims import ml as xml
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import sys
import xgboost

from xifaims import processing as xp
from xifaims import plots as xpl
from xifaims import ml
import shap
import matplotlib.pyplot as plt
import pickle
import os



def summarize_df(all_clf):
    # summarize data in dataframe
    res_df = {"clf": [], "set": [], "pearson": [], "r2": [], "mse": []}
    for ii, grpi in all_clf.groupby(["classifier", "Set"]):
        res_df["clf"].append(ii[0])
        res_df["set"].append(ii[1])
        res_df["pearson"].append(pearsonr(grpi["CV_Train"], grpi["CV_Predict"])[0])
        res_df["r2"].append(pearsonr(grpi["CV_Train"], grpi["CV_Predict"])[0] ** 2)
        res_df["mse"].append(mean_squared_error(grpi["CV_Train"], grpi["CV_Predict"]))

    res_df = pd.DataFrame(res_df)
    res_df = res_df.round(2)
    res_df["run"] = prefix
    res_df = res_df.sort_values(by=["clf", "set", "pearson"])
    return res_df


# %%
# setup
# scale = True
# nonunique = True
# charge = "all"
# prefix = "all_features"
# exclude = []
# include = []

# # parsing and options
results_dir = "results_dev"
infile_loc = "data/combined_8PMLunique_4PMLS_nonu.csv"
config_loc = "parameters/faims_structure_selection.yaml"

# %%
prefix = os.path.basename(config_loc.split(".")[0]) + "-" + os.path.basename(infile_loc.replace(".csv", ""))
config = yaml.load(open(config_loc), Loader=yaml.FullLoader)
config["grid"] = "small"
config["jobs"] = -1
print(config)
dir_res = os.path.join(results_dir, prefix)
if not os.path.exists(dir_res):
    os.makedirs(dir_res)
############################################################

# read file and annotate CV
df = pd.read_csv(infile_loc)

# set cv
df["CV"] = - df["run"].apply(xp.get_faims_cv)

# remove non-unique
df = xp.preprocess_nonunique(df)
# split into targets and decoys
df_TT, df_DX = xp.split_target_decoys(df)

# filter by charge
df_TT = xp.charge_filter(df_TT, config["charge"])
df_DX = xp.charge_filter(df_DX, config["charge"])

# compute features
df_TT_features = xf.compute_features(df_TT)
df_DX_features = xf.compute_features(df_DX)

# drop_features = ["proline", "DE", "KR", "log10mass", "Glycine"]
df_TT_features = df_TT_features.drop(config["exclude"], axis=1)
df_DX_features = df_DX_features.drop(config["exclude"], axis=1)

# only filter if include is specified, else just take all columns
if len(config["include"]) > 0:
    df_TT_features = df_TT_features[config["include"]]
    df_DX_features = df_DX_features[config["include"]]


# regression
print("XGB Regression ...")
xgb_options = {"grid": config["grid"], "jobs": config["jobs"], "type": "XGBR"}
xgbr_predictions, xgbr_metric, xgbr_gs, xgbr_clf = xml.training(df_TT, df_TT_features, model="XGB",
                                                                scale=True, model_args=xgb_options)

xp.store_for_shap(df_TT, df_TT_features, df_DX, df_DX_features, xgbr_clf, path="shap", prefix="xgbr_structure_")

#%%
from sklearn.model_selection import StratifiedKFold
X = df_TT_features
y = df_TT["CV"]
skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(X, y)

pearson = []
names = np.tile(["Train", "Predict"], 3)
predictions_df = pd.DataFrame(index=X.index)
predictions_df["Predicted CV"] = 100
predictions_df["CV-FOLD"] = -1
predictions_df["Observed CV"] = 100
cc = 1
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    tmp_clf = xgboost.XGBRegressor(**xgbr_gs.best_params_)
    xgb_clf = tmp_clf.fit(X_train, y_train)
    xtrain_hat = xgb_clf.predict(X_train)
    xtest_hat = xgb_clf.predict(X_test)
    pearson.append(pearsonr(xtrain_hat, y_train)[0])
    pearson.append(pearsonr(xtest_hat, y_test)[0])
    predictions_df["Predicted CV"].iloc[test_index] = xtest_hat
    predictions_df["Observed CV"].iloc[test_index] = y_test
    predictions_df["CV-FOLD"].iloc[test_index] = cc
    cc += 1

res_df = pd.DataFrame(pearson).transpose()
res_df.columns = names
res_df = res_df.transpose().reset_index().round(2)
res_df.columns = ["setting", "pearsonr"]
print(res_df)
predictions_df_decoys = pd.DataFrame(xgb_clf.predict(df_DX_features))
predictions_df_decoys.columns = ["Predicted CV"]
predictions_df_decoys["Observed CV"] = df_DX["CV"]
predictions_df_decoys["CV-FOLD"] = -1
predictions_df["Type"] = "TT"
predictions_df_decoys["Type"] = "DX"
concat_df = pd.concat([predictions_df, predictions_df_decoys])
g = sns.jointplot(x="Observed CV", y="Predicted CV", data=concat_df, hue="Type")
g.ax_joint._axes.plot([-80, -10], [-80, -10], c="k", lw="2", alpha=0.7, ls="--")
plt.show()

concat_df["error"] = concat_df["Observed CV"] - concat_df["Predicted CV"]
f, ax = plt.subplots(1, figsize=(1, 4))
#ax = sns.histplot(data=concat_df, x="error", hue="Type", stat="density", element="step", common_norm=True)
ax = sns.boxplot(data=concat_df, y="error", x="Type")
ax.axhline(0.0, lw=2, c="k", alpha=0.7, zorder=-1)
ax.set(ylabel="CV prediction error")
sns.despine(ax=ax)
plt.savefig("notebooks/TT_TD_prediction_error.png")
plt.savefig("notebooks/TT_TD_prediction_error.svg")
plt.show()