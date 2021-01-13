import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from xirt import sequences
import glob
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
shap.initjs()


def stats_from_errors():
    """
    Do anove on the errors by goruping by charge state. Performs anova and follow.up test.
    """
    anova_result = f_oneway(*[VAL_meta[VAL_meta["exp charge"] == i]["error"].values for i in
                              VAL_meta["exp charge"].unique()])
    print("Anova:", anova_result)

    print(VAL_meta["exp charge"].value_counts())
    # perform multiple pairwise comparison (Tukey HSD)
    m_comp = pairwise_tukeyhsd(endog=VAL_meta['error'], groups=VAL_meta['exp charge'], alpha=0.05)
    tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns = m_comp._results_table.data[0])
    tukey_data = tukey_data.sort_values("p-adj")
    print(tukey_data)


#data_dic = pickle.load(open(r"..\results\xifaims_8PM4PM_CO_NAVG_small.p", "rb"))
#data_dic = pickle.load(open(r"..\results\xifaims_8PM4PM_CO_AVG_huge.p", "rb"))
#data_dic = pickle.load(open(r"..\results\xifaims_8PM4PM_CO_NAVG_huge.p", "rb"))
data_dic = pickle.load(open(r"results\xifaims_8PM4PM_CO_NAVG_huge_nofeats.p", "rb"))
data_dic.keys()

metrics_df = data_dic["metrics"].round(2).set_index("split")
metrics_df

predictions_df = data_dic["predictions_df"]
predictions_df_val = predictions_df[predictions_df["Split"] == "Test"]
predictions_df_val["error"] = predictions_df_val["predictions"] - predictions_df_val["observed"]

TT_meta, TT_features, TT_predictions = data_dic["data"]["TT_train"]
VAL_meta, VAL_features, VAL_predictions = data_dic["data"]["TT_val"]
VAL_meta["error"] = predictions_df_val["error"].values
VAL_meta = VAL_meta.reset_index()

z3 = VAL_meta[VAL_meta["exp charge"] == 3].index
z3_X = VAL_meta["exp charge"] == 3

z4 = VAL_meta[VAL_meta["exp charge"] == 4].index
z4_X = VAL_meta["exp charge"] == 4

#%%
print("Loading model, train, val, dx features and best features")
# model
model = data_dic["xgb"]
print(data_dic["data"].keys())

# data consists of tuples with meta data and feature data
all_data = data_dic["data"]
TT_train, TT_train_features, TT_predictions = all_data["TT_train"]
TT_val, TT_val_features, TT_val_predictions = all_data["TT_val"]
DX = all_data["DX"]
# print(all_data)

# use shortcut
features = data_dic["best_features_gs"]
shap_feature_z3 = features  == "p.charge"
X = TT_val_features[features]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
#shap_values_inter = shap.TreeExplainer(model).shap_interaction_values(X)
stats_from_errors()
plots()

def plots():
    shap.summary_plot(shap_values[z3.values], X.reset_index(drop=True).loc[z3.values], show=False)
    plt.show()

    shap.summary_plot(shap_values[z4.values], X.reset_index(drop=True).loc[z4.values], show=False)
    plt.show()

    f, ax = plt.subplots(1, figsize=(4, 4))
    ax = sns.boxplot(x="exp charge", y="error", data=VAL_meta, palette="rocket_r")
    ax.axhline(0, lw=2, c="k", zorder=-1)
    plt.show()




def analyze_charge_3(TT_val, shap_values, shap_feature_z3):
    sns.set(palette="deep", style="white", context="talk")
    from sklearn.cluster import KMeans
    from pyteomics import parser
    TT_val_z3 = TT_val[TT_val["match charge"] == 3]
    # get only the shap values for z=3 and only the corresponding psms
    color = KMeans(2).fit_predict(shap_values[z3.values, shap_feature_z3].reshape(-1, 1))

    f,ax = plt.subplots()
    ax.scatter([3] * 123, shap_values[z3.values, shap_feature_z3], c=color)
    ax.set(xlabel="charge", ylabel="SHAP value")
    plt.show()
    TT_val_z3["SHAP3_kmeans"] = color
    TT_val_z3["SHAP"] = shap_values[z3.values, shap_feature_z3]

    TT_val_z3_melt = TT_val_z3.melt(value_vars=["PeptideLength1", "PeptideLength2", "exp m/z",
                                                "deltaScore", "exp mass", "SHAP"], id_vars=["SHAP3_kmeans"])

    sns.catplot(x="SHAP3_kmeans", y="value", col="variable", data=TT_val_z3_melt,
                kind="box", sharey=False, col_wrap=3)
    plt.show()


    modx_format = TT_val_z3["seq1seq2"].apply(sequences.rewrite_modsequences)
    alphabet = sequences.get_alphabet(modx_format)
    aa_counts = [pd.Series(parser.amino_acid_composition(i, labels=alphabet)) for i in modx_format]
    aa_counts = pd.DataFrame(aa_counts)
    aa_counts = aa_counts.fillna(0)


    aa_counts_f = aa_counts[aa_counts.columns[np.where(aa_counts.sum(axis=0) > 10)]]
    aa_counts_f = aa_counts_f.div(aa_counts_f.sum(axis=1), axis=0).round(2)
    aa_counts_f["shap"] = color
    aa_counts_melt_f = aa_counts_f.melt(id_vars=["shap"])

    aa_counts = aa_counts.div(aa_counts.sum(axis=1), axis=0).round(2)
    aa_counts["shap"] = color
    aa_counts_melt = aa_counts.melt(id_vars=["shap"])

    order1 = aa_counts_melt.groupby(["shap", "variable"]).agg(np.mean).loc[0].sort_values(by="value", ascending=False).index.values
    f, ax = plt.subplots(1, figsize=(14, 6))
    ax = sns.barplot(x="variable", y="value", hue="shap", data=aa_counts_melt, ax=ax,
                     order=order1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,
                       horizontalalignment='right')
    plt.show()

    order2 = aa_counts_melt_f.groupby(["shap", "variable"]).agg(np.mean).loc[0].sort_values(by="value", ascending=False).index.values
    f, ax = plt.subplots(1, figsize=(14, 6))
    ax = sns.barplot(x="variable", y="value", hue="shap", data=aa_counts_melt_f, ax=ax,
                     order=order2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,
                       horizontalalignment='right')
    plt.show()