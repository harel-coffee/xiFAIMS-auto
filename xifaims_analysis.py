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
from xifaims import processing as xp
from xifaims import features as xf
from xifaims import plots as xpl
from xifaims import ml as xml
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def summarize_df(all_clf):
    # summarize data in dataframe
    res_df = {"clf": [], "set": [], "pearson": [], "r2": [], "mse": []}
    for ii, grpi in all_clf.groupby(["classifier", "Set"]):
        res_df["clf"].append(ii[0])
        res_df["set"].append(ii[1])
        res_df["pearson"].append(pearsonr(grpi["CV_Train"], grpi["CV_Predict"])[0])
        res_df["r2"].append(pearsonr(grpi["CV_Train"], grpi["CV_Predict"])[0]**2)
        res_df["mse"].append(mean_squared_error(grpi["CV_Train"], grpi["CV_Predict"]))
    
    res_df = pd.DataFrame(res_df)
    res_df = res_df.round(2)
    res_df["run"] = prefix
    res_df = res_df.sort_values(by=["clf", "set", "pearson"])
    return res_df
#%%
# setup
# scale = True
# nonunique = True
# charge = "all"
# prefix = "all_features"
# exclude = []
# include = []

config_loc = sys.argv[1]
infile_loc = sys.argv[2]

# # parsing and options
# infile_loc = "data/4PM_DSS_LS_nonunique1pCSM.csv"
# config_loc = "parameters/faims_all.yaml"
# config_loc = "parameters/faims_minimal.yaml"
# config_loc = "parameters/faims_structure.yaml"

#%%
prefix = os.path.basename(config_loc.split(".")[0]) + "-" + os.path.basename(infile_loc.replace(".csv", ""))
config = yaml.load(open(config_loc), Loader=yaml.FullLoader)
config["grid"] = "large"
config["jobs"] = 40
print(config)
dir_res = os.path.join("results", prefix)
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

df_TT_features = df_TT_features[config["include"]]
df_DX_features = df_DX_features[config["include"]]

xpl.feature_correlation_plot(df_TT_features, dir_res, prefix="TT_")
#%%
# train baseline
# classifier
print("SVM ...")
svm_options = {"jobs": config["jobs"], "type": "SVC"}
svm_predictions, svm_metric, svm_gs, svm_clf = xml.training(df_TT, df_TT_features, model="SVM",
                                                            scale=True, model_args=svm_options)

# regression
print("SVR ...")
svm_options = {"jobs": config["jobs"], "type": "SVR"}
svr_predictions, svr_metric, svr_gs, svr_clf = xml.training(df_TT, df_TT_features, model="SVM",
                                                            scale=True, model_args=svm_options)

# regression
print("XGB Regression ...")
xgb_options = {"grid": config["grid"], "jobs": config["jobs"], "type": "XGBR"}
xgbr_predictions, xgbr_metric, xgbr_gs, xgbr_clf = xml.training(df_TT, df_TT_features, model="XGB",
                                                            scale=True, model_args=xgb_options)
# classification
print("XGB classification ...")
xgb_options = {"grid": config["grid"], "jobs": config["jobs"], "type": "XGBC"}
xgbc_predictions, xgbc_metric, xgbc_gs, xgbc_clf = xml.training(df_TT, df_TT_features, model="XGB",
                                                            scale=True, model_args=xgb_options)

# classification
print("FNN ...")
FNN_options = {"grid": config["grid"]}
FNN_predictions, fnn_metric, fnn_gs, fnn_clf = xml.training(df_TT, df_TT_features, model="FNN",
                                                            scale=True, model_args=FNN_options)

# %% organize results
# concat to one dataframe for easier plotting
all_clf = pd.concat([svm_predictions, svr_predictions, xgbr_predictions, xgbc_predictions, FNN_predictions])
all_clf["run"] = prefix
all_clf["config"] = config_loc
all_clf["infile"] = infile_loc
all_clf["run"] = prefix

all_metrics = pd.concat([svm_metric,svr_metric, xgbr_metric, xgbc_metric, fnn_metric])
all_metrics["config"] = config_loc
all_metrics["infile"] = infile_loc
all_metrics["run"] = prefix

# summarize data
res_df = summarize_df(all_clf)
# %% store data
all_clf.to_csv(os.path.join(dir_res, "{}_summary_CV.csv".format(prefix)))
all_metrics.to_csv(os.path.join(dir_res, "{}_all_metrics.csv".format(prefix)))
res_df.to_csv(os.path.join(dir_res, "{}_summary_predictions.csv".format(prefix)))
print(config)
print(dir_res)
print("Done.")

# print ("QC Plots")
# xpl.train_test_scatter_plot(all_clf, dir_res)
# xpl.cv_performance_plot(all_metrics, dir_res)


feature_important = xgbr_clf.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.plot(kind='barh')
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(dir_res, "{}_feature_importance.png".format(prefix)))
plt.clf()