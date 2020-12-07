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
data_dic = pickle.load(open(r"results\xifaims_8PM4PM_CO_NAVG_huge.p", "rb"))

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
predictions_df_val = predictions_df[predictions_df["Split"] == "Test"]
predictions_df_val["error"] = predictions_df_val["predictions"] - predictions_df_val["observed"]

TT_meta, TT_features = data_dic["data"]["TT_train"]
VAL_meta, VAL_features = data_dic["data"]["TT_val"]
VAL_meta["error"] = predictions_df_val["error"].values
VAL_meta = VAL_meta.reset_index()

z3 = VAL_meta[VAL_meta["exp charge"] == 3].index
z3_X = VAL_meta["exp charge"] == 3

z4 = VAL_meta[VAL_meta["exp charge"] == 4].index
z4_X = VAL_meta["exp charge"] == 4

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
    #shap_values_inter = shap.TreeExplainer(model).shap_interaction_values(X)

    shap.summary_plot(shap_values[z3.values], X.reset_index(drop=True).loc[z3.values], show=False)
    plt.show()

    shap.summary_plot(shap_values[z4.values], X.reset_index(drop=True).loc[z4.values], show=False)
    plt.show()

    f, ax = plt.subplots(1, figsize=(4, 4))
    ax = sns.boxplot(x="exp charge", y="error", data=VAL_meta, palette="rocket_r")
    ax.axhline(0, lw=2, c="k", zorder=-1)
    plt.show()

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from scipy.stats import f_oneway
    anova_result = f_oneway(*[VAL_meta[VAL_meta["exp charge"] == i]["error"].values for i in
                              VAL_meta["exp charge"].unique()])
    print("Anova:", anova_result)

    print(VAL_meta["exp charge"].value_counts())
    # perform multiple pairwise comparison (Tukey HSD)
    m_comp = pairwise_tukeyhsd(endog=VAL_meta['error'], groups=VAL_meta['exp charge'], alpha=0.05)
    tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns = m_comp._results_table.data[0])
    tukey_data = tukey_data.sort_values("p-adj")
    print(tukey_data)
