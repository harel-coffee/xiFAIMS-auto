import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import glob
shap.initjs()
#data_dic = pickle.load(open(r"..\results\xifaims_8PM4PM_CO_NAVG_small.p", "rb"))
#data_dic = pickle.load(open(r"..\results\xifaims_8PM4PM_CO_AVG_huge.p", "rb"))
#data_dic = pickle.load(open(r"..\results\xifaims_8PM4PM_CO_NAVG_huge.p", "rb"))
data_dic = pickle.load(open(r"results\xifaims_8PM4PM_OH_NAVG_huge.p", "rb"))

#data_dic = pickle.load(open(r"..\results\xifaims_8PM4PM_OH_AVG_tiny.p", "rb"))

data_dic.keys()
#dict_keys(['best_features_gs', 'summary_gs', 'best_params_gs', 'xifaims_params', 'xifaims_config', 'xgb', 'predictions_df', 'data', 'metrics', 'df_unique', 'df_nonunique'])

# takes very long?!
# get the summary metric data from all pickle files
# files = glob.glob(r"results\*.p")
# data_overview = []
# for i in files:
#    df_tmp = pickle.load(open(i, "rb"))["metrics"]
#    df_tmp["file"] = i
#    data_overview.append(df_tmp)
# df_data_overview = pd.concat(data_overview)
# df_data_overview = df_data_overview[df_data_overview["split"] == "Validation"]
# df_data_overview = df_data_overview.sort_values("pearsonr", ascending=False)

metrics_df = data_dic["metrics"].round(2).set_index("split")
metrics_df

predictions_df = data_dic["predictions_df"]
predictions_df_val = predictions_df[predictions_df["Split"] == "Val"]

TT_meta, TT_features = data_dic["data"]["TT_train"]
VAL_meta, VAL_features = data_dic["data"]["TT_val"]
VAL_meta = VAL_meta.reset_index()

z3 = VAL_meta[VAL_meta["exp charge"] == 3].index

def shap_analysis():
    print("Loading model, train, val, dx features and best features")
    # model
    model = data_dic["xgb"]
    print(data_dic["data"].keys())

    # data consists of tuples with meta data and feature data
    all_data = data_dic["data"]
    TT_train, TT_train_features = all_data["TT_train"]
    TT_val, TT_val_features = all_data["TT_val"]
    DX = all_data["DX"]
    # print(all_data)

    # use shortcut
    features = data_dic["best_features_gs"]
    X = TT_val_features[features]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_values_inter = shap.TreeExplainer(model).shap_interaction_values(X)