# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:56:23 2020

@author: hanjo
"""
import os
import pandas as pd

from xifaims import processing as xp
from xifaims import features as xf
from xifaims import plots as xpl
from xifaims import ml as xml

# setup
scale = True
nonunique = True
charge = "all"
prefix = "all_features"
exclude = []
include = []
infile = "data/4PM_DSS_LS_nonunique1pCSM.csv"
prefix = "test_run"

dir_res = os.path.join("results", prefix)
if not os.path.exists(dir_res):
    os.makedirs(dir_res)
###################################################################################################

# read file and annotate CV
df = pd.read_csv(infile)

# set cv
df["CV"] = - df["run"].apply(xp.get_faims_cv)

# remove non-unique
df = xp.preprocess_nonunique(df)
# split into targets and decoys
df_TT, df_DX = xp.split_target_decoys(df)

# filter by charge
df_TT = xp.charge_filter(df_TT, charge)
df_DX = xp.charge_filter(df_DX, charge)

# compute features
df_TT_features = xf.compute_features(df_TT)
df_DX_features = xf.compute_features(df_DX)
drop_features = ["proline", "DE", "KR", "log10mass", "Glycine"]
df_TT_features = df_TT_features.drop(drop_features, axis=1)
df_DX_features = df_DX_features.drop(drop_features, axis=1)

xpl.feature_correlation_plot(df_TT_features, dir_res, prefix="TT_")

# train baseline
svm_options = {"jobs": 8}
svm_predictions, svm_metric, svm_gs, svm_clf = xml.training(df_TT, df_TT_features, model="SVR", 
                                                            scale=True, model_args=svm_options)

xgb_options = {"grid": "tiny", "jobs": 8}
xgb_predictions, xgb_metric, xgb_gs, xgb_clf = xml.training(df_TT, df_TT_features, model="XGB", 
                                                            scale=True, model_args=xgb_options)

FNN_options = {}
FNN_predictions, fnn_metric, fnn_gs, fnn_clf = xml.training(df_TT, df_TT_features, model="FNN", 
                                                            scale=True, model_args=FNN_options)
# concat to one dataframe for easier plotting
all_clf = pd.concat([svm_predictions, xgb_predictions, FNN_predictions])
all_clf["run"] = prefix
all_metrics = pd.concat([svm_metric, xgb_metric])

print ("QC Plots")
xpl.train_test_scatter_plot(all_clf, dir_res)
xpl.cv_performance_plot(all_metrics, dir_res)
#%%
#
# summarize data in dataframe
res_df = {"clf": [], "set": [], "pearson": [], "r2": [], "mse": []}
for ii, grpi in all_clf.groupby(["classifier", "Set"]):
    res_df["clf"].append(ii[0])
    res_df["set"].append(ii[1])
    res_df["pearson"].append(pearsonr(grpi["CV_Train"], grpi["CV_Predict"])[0])
    res_df["r2"].append(r2_score(grpi["CV_Train"], grpi["CV_Predict"]))
    res_df["mse"].append(mean_squared_error(grpi["CV_Train"], grpi["CV_Predict"]))

res_df = pd.DataFrame(res_df)
res_df = res_df.round(2)
res_df["run"] = prefix
res_df = res_df.sort_values(by=["clf", "set", "pearson"])
print(res_df)

all_metrics["run"] = prefix
all_metrics.to_csv(os.path.join(
    dir_res, "{}_all_metrics.csv".format(prefix)))
res_df.to_csv(os.path.join(
    dir_res, "{}_summary_predictions.csv".format(prefix)))
all_clf.to_csv(os.path.join(dir_res, "{}_summary_CV.csv".format(prefix)))