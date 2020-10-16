# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:32:59 2020

@author: hanjo
"""

from xifaims import optimizer as xo
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
from xifaims import processing as xp
from xirt.features import get_pi
from xirt import xirtnet
from xirt import predictor as xr
from xirt import processing as xirtp
from xirt import __main__ as xm
import sys
import os
sys.path.append("../")  # go to parent dir
optimizer = xo.AdaBeliefOptimizer(0.001)


def normalize(array):
    """Standardize array values."""
    return(array - array.mean()) / array.std()


def converter(prediction, ys, xirt_loc):
    """Convenient transformation of regression / ordinal output for xirtnet."""
    if "ordinal" in xirt_loc:
        preds = xr.sigmoid_to_class(prediction[0])
        obs = xr.sigmoid_to_class(ys[0])
    else:
        preds = np.ravel(prediction[0])
        obs = ys[0]
    return preds, obs


# test:
# epochs, batch_size, weights
# transfer learning

if len(sys.argv) != 6:
    print("""Please supply two arguments with the script: 
          python xirt_analysis.py psms.csv xirt.yaml learning.yaml""")

print(sys.argv)

infile = sys.argv[1]
xirt_loc = sys.argv[2]
setup_loc = sys.argv[3]
batch_size = int(sys.argv[4])
epochs = int(sys.argv[5])

# infile = "data/4PM_DSS_LS_nonunique1pCSM.csv"
# xirt_loc = "parameters/xirt_faims_ordinal_aux.yaml"
# setup_loc = "parameters/xirt_learning.yaml"

basename = os.path.basename(xirt_loc).replace(".yaml", "")
out_dir = os.path.join("results", 'xirt', basename)
out_dir_root = os.path.join("results", 'xirt')

if not os.path.exists(out_dir_root):
    os.makedirs(out_dir_root)

# sys.exit()
# infile = "../data/4PM_DSS_LS_nonunique1pCSM.csv"
# infile = "data/8PM_DSS_Liu_nonunique1pCSM.csv"
# infile = "data/4PM_DSS_LS_nonunique1pCSM.csv"
# infile = "data/293T_DSSO_nonunique1pCSM.csv"
# infile = "data/26S_BS3_LS_nonunique1pCSM.csv"


xirt_params = yaml.load(open(xirt_loc), Loader=yaml.FullLoader)
xirt_params["learning"]["epochs"] = epochs
xirt_params["learning"]["batch_size"] = batch_size
xirt_params["callbacks"]["callback_path"] = out_dir

learning_params = yaml.load(open(setup_loc), Loader=yaml.FullLoader)
matches_df = pd.read_csv(infile)
if "LS" in infile:
    matches_df["cv"] = matches_df["run"].apply(xp.get_faims_cv, args=("LS",))
elif "Liu" in infile:
    matches_df["cv"] = matches_df["run"].apply(xp.get_faims_cv, args=("Liu",))

matches_df = matches_df.rename(columns={"Score": "score", "PepSeq1": "Peptide1",
                                        "PepSeq2": "Peptide2", "fdr": "FDR"})
print(np.unique(matches_df["cv"]))
print(len(np.unique(matches_df["cv"])))
print(matches_df.head())


# %%
training_data = xr.preprocess(matches_df, sequence_type="crosslink",
                              max_length=-1, cl_residue=True, fraction_cols=["cv"])
training_data.set_fdr_mask(fdr_cutoff=0.05, str_filter="")
training_data.psms["pi"] = training_data.psms["PepSeq1PepSeq2_str"].apply(get_pi)
# training_data.psms["tmass"] = (training_data.psms["exp mass"]
#                                - training_data.psms["exp mass"].mean()) / training_data.psms["exp mass"].std()
# training_data.psms["tmz"] = (training_data.psms["exp m/z"]
#                              - training_data.psms["exp m/z"].mean()) / training_data.psms["exp m/z"].std()
# #training_data.psms["tmz"] = training_data.psms["exp m/z"] / training_data.psms["exp m/z"].max()

training_data.psms["pi"] = normalize(training_data.psms["pi"])
training_data.psms["tmass"] = normalize(training_data.psms["exp mass"])
training_data.psms["tmz"] = normalize(training_data.psms["exp m/z"])

# %%
# init the network
# pip install jupyter-kite
xirtnetwork = xirtnet.xiRTNET(xirt_params, input_dim=training_data.features1.shape[1])
frac_cols = sorted([xirtnetwork.output_p[tt.lower() + "-column"]
                    for tt in xirt_params["predictions"]["fractions"]])
cont_cols = sorted([xirtnetwork.output_p[tt.lower() + "-column"]
                    for tt in xirt_params["predictions"]["continues"]])
all_cols = np.concatenate([frac_cols, cont_cols])
print(frac_cols)
print(cont_cols)

# do CV
training_data.set_unique_shuffled_sampled_training_idx()
n_splits = 3
test_size = 0.1
cv_counter = 0
results = {"r2": [], "pearsonr": [], "split": [], "yaml": []}

for train_idx, val_idx, pred_idx in training_data.iter_splits(n_splits=n_splits,
                                                              test_size=test_size):
    print(cv_counter)
    print(len(train_idx))
    print(len(val_idx))
    cv_counter += 1
    xt_cv = training_data.get_features(train_idx)
    yt_cv = training_data.get_classes(train_idx, frac_cols=frac_cols, cont_cols=cont_cols)

    xv_cv = training_data.get_features(val_idx)
    yv_cv = training_data.get_classes(val_idx, frac_cols=frac_cols, cont_cols=cont_cols)

    xp_cv = training_data.get_features(pred_idx)
    yp_cv = training_data.get_classes(pred_idx, frac_cols=frac_cols, cont_cols=cont_cols)
    
    xirtnetwork.build_model(siamese=xirt_params["siamese"]["use"])
    xirtnetwork.compile()
    callbacks = xirtnetwork.get_callbacks(suffix=str(cv_counter).zfill(2))

    # # get parameters from config file
    # loss = {i: xirtnetwork.output_p[i + "-loss"] for i in xirtnetwork.tasks}
    # metric = {i: xirtnetwork.output_p[i + "-metrics"] for i in xirtnetwork.tasks}
    # loss_weights = {i: xirtnetwork.output_p[i + "-weight"] for i in xirtnetwork.tasks}
    # xirtnetwork.model.compile(loss=loss, optimizer=optimizer, metrics=metric, loss_weights=loss_weights)
    # xirtnetwork.model.summary()

    history = xirtnetwork.model.fit(xt_cv, yt_cv, validation_data=(xv_cv, yv_cv),
                                    epochs=xirt_params["learning"]["epochs"],
                                    batch_size=xirt_params["learning"]["batch_size"],
                                    verbose=1, callbacks=callbacks)

    # aux has multiple output variables which requires looping for plotting
    if "aux" in xirt_loc:
        predictions_v = xirtnetwork.model.predict(xv_cv)
        predictions_t = xirtnetwork.model.predict(xt_cv)
        predictions_p = xirtnetwork.model.predict(xp_cv)

        predictions_vv, obs_v = converter(predictions_v, yv_cv, xirt_loc)
        predictions_tt, obs_t = converter(predictions_t, yt_cv, xirt_loc)
        predictions_pp, obs_p = converter(predictions_p, yp_cv, xirt_loc)
        
        predictions_v[0] = predictions_vv
        predictions_t[0] = predictions_tt
        predictions_p[0] = predictions_pp
        yv_cv[0] = obs_v
        yt_cv[0] = obs_t
        yp_cv[0] = obs_p

        # store for later
        results["pearsonr"].append(pearsonr(predictions_vv, obs_v)[0])
        results["r2"].append(results["pearsonr"][-1]**2)

        results["pearsonr"].append(pearsonr(predictions_tt, obs_t)[0])
        results["r2"].append(results["pearsonr"][-1]**2)

        results["pearsonr"].append(pearsonr(predictions_pp, obs_p)[0])
        results["r2"].append(results["pearsonr"][-1]**2)
        
        f, ax = plt.subplots(3, len(all_cols), figsize=(4*len(all_cols), 8))
        for ii, col in enumerate(all_cols):
            prs = np.round(pearsonr(np.ravel(predictions_t[ii]), yt_cv[ii])[0], 2)
            ax[0][ii].scatter(np.ravel(predictions_t[ii]), yt_cv[ii])
            ax[0][ii].set(xlabel="Predicted\nTraining", ylabel="Observerd",
                          title=col + " prs: {}".format(prs))

            prs = np.round(pearsonr(np.ravel(predictions_v[ii]), yv_cv[ii])[0], 2)
            ax[1][ii].scatter(np.ravel(predictions_v[ii]), yv_cv[ii])
            ax[1][ii].set(xlabel="Predicted\nValidation", ylabel="Observerd",
                          title=col + " prs: {}".format(prs))
            
            prs = np.round(pearsonr(np.ravel(predictions_p[ii]), yp_cv[ii])[0], 2)
            ax[2][ii].scatter(np.ravel(predictions_p[ii]), yp_cv[ii])
            ax[2][ii].set(xlabel="Predicted\nPrediction", ylabel="Observerd",
                          title=col + " prs: {}".format(prs))            

    else:
        # single task
        predictions_v = xirtnetwork.model.predict(xv_cv)
        predictions_t = xirtnetwork.model.predict(xt_cv)
        predictions_p = xirtnetwork.model.predict(xp_cv)

        predictions_v, obs_v = converter([predictions_v], yv_cv, xirt_loc)
        predictions_t, obs_t = converter([predictions_t], yt_cv, xirt_loc)
        predictions_p, obs_p = converter([predictions_p], yp_cv, xirt_loc)
        
        # validation
        results["pearsonr"].append(pearsonr(predictions_v, obs_v)[0])
        results["r2"].append(results["pearsonr"][-1]**2)

        # train
        results["pearsonr"].append(pearsonr(predictions_t, obs_t)[0])
        results["r2"].append(results["pearsonr"][-1]**2)

        # train
        results["pearsonr"].append(pearsonr(predictions_p, obs_p)[0])
        results["r2"].append(results["pearsonr"][-1]**2)
        
        f, ax = plt.subplots(2, len(all_cols), figsize=(4*len(all_cols), 8))
        # train
        prs = np.round(results["pearsonr"][-1], 2)
        ax[0].scatter(predictions_t, obs_t)
        ax[0].set(xlabel="Predicted\nTraining", ylabel="Observerd", title="cv prs: {}".format(prs))
        # validation
        prs = np.round(results["pearsonr"][-2], 2)
        ax[1].scatter(predictions_v, obs_v)
        ax[1].set(xlabel="Predicted\nValidation",
                  ylabel="Observerd", title="cv prs: {}".format(prs))

        prs = np.round(results["pearsonr"][-2], 2)
        ax[2].scatter(predictions_v, obs_v)
        ax[2].set(xlabel="Predicted\nPrediction",
                  ylabel="Observerd", title="cv prs: {}".format(prs))
        
    results["split"].append("validation")
    results["split"].append("training")
    results["split"].append("prediction")
    results["yaml"].extend([basename]*3)

    sns.despine()
    plt.tight_layout()
    plt.savefig(out_dir + "_performance_cv_{}".format(str(cv_counter).zfill(2)))
    plt.clf()
res_df = pd.DataFrame(results)
res_df = res_df.round(3)
res_df.to_csv(out_dir + "_performance_summary.csv")

# %%
# import shap
# sampleidx = xt_cv[0].sample(100).index
# bg = np.array((xt_cv[0].loc[sampleidx], xt_cv[1].loc[sampleidx]))
# e = shap.DeepExplainer(xirtnetwork.model, bg)
# shap_values = e.shap_values(x_test[1:5])

# try:
#     x = pd.DataFrame()
#     #x["CV_Predictions"] = np.ravel(predictions[0])
#     x["CV_Predictions"] = np.ravel(predictions)
#     x["CV_Obs"] = yt_cv[0]
# except:
#     x = pd.DataFrame()
#     x["CV_Predictions"] = xr.sigmoid_to_class(predictions[0])
#     x["CV_Obs"] = xr.sigmoid_to_class(yt_cv[0])

# sns.jointplot("CV_Predictions", "CV_Obs", data=x, kind="hist")
# print(xirt_loc)
# print (r2_score(x["CV_Predictions"], x["CV_Obs"]))
# print (pearsonr(x["CV_Predictions"], x["CV_Obs"]))

# try:
#     x["pi_Predictions"] = np.ravel(predictions[1])
#     x["pi_Obs"] = yt_cv[1]
#     sns.jointplot("pi_Predictions", "pi_Obs", data=x, kind="hist")

# except:
#     pass
