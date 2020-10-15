# # -*- coding: utf-8 -*-
# """
# Created on Wed Oct 14 23:33:26 2020

# @author: hanjo
# """
# import shap
# # %% xgb transform
#    mybooster = xgbr.get_booster()
#    model_bytearray = mybooster.save_raw()[4:]

#    def myfun(self=None):
#    return model_bytearray

#    mybooster.save_raw = myfun

#    # %% SHAP
#    explainer = shap.TreeExplainer(xgbr)
#    shap_values = explainer.shap_values(features_df.loc[training_idx])
#    shap_interaction_values = explainer.shap_interaction_values(
#    features_df.loc[training_idx])

#    shap.summary_plot(shap_values, train_df)
#    shap.summary_plot(shap_values, train_df, plot_type="bar")

#    # force plot
#    shap.force_plot(explainer.expected_value, shap_values[0, :],
#            features_df.iloc[0, :], matplotlib=True, show=False)
#    plt.tight_layout()
#    plt.savefig(os.path.join(
#    dir_res, "{}_force_plot_sample0.png".format(prefix)))
#    plt.clf()

#    # waterfall plot
#    for nn in [0, 5, 10, 100, 500]:
#    shap.waterfall_plot(explainer.expected_value, shap_values[nn, :],
#                    features_df.iloc[nn, :], max_display=20, show=False)
#    plt.title("observed cv: {}".format(df["CV"].iloc[nn]))
#    plt.tight_layout()
#    plt.savefig(os.path.join(
#    dir_res, "{}_waterfall_sample{}.png".format(prefix, nn)))
#    plt.clf()

#    # dependence plot
#    shap.dependence_plot("mass", shap_values, train_df, show=False)
#    plt.tight_layout()
#    plt.savefig(os.path.join(dir_res, "{}_mass_dependence.png".format(prefix)))
#    plt.clf()

#    # dependance plot mass and charge
#    shap.dependence_plot("mass", shap_values, train_df,
#                 show=False, interaction_index="p.charge")
#    plt.tight_layout()
#    plt.savefig(os.path.join(
#    dir_res, "{}_mass_charge_dependence.png".format(prefix)))
#    plt.clf()

#    # summary compact
#    shap.summary_plot(shap_values, train_df, show=False)
#    plt.tight_layout()
#    plt.savefig(os.path.join(
#    dir_res, "{}_summary_plot_compact.png".format(prefix)))
#    plt.clf()

#    # summary bar
#    shap.summary_plot(shap_values, train_df, plot_type="bar", show=False)
#    plt.tight_layout()
#    plt.savefig(os.path.join(
#    dir_res, "{}_summary_plot_bar.png".format(prefix)))
#    plt.clf()

#    # summary bar
#    shap.summary_plot(shap_values, train_df, plot_type="bar", show=False)
#    plt.tight_layout()
#    plt.savefig(os.path.join(
#    dir_res, "{}_summary_interaction_bar.png".format(prefix)))
#    plt.clf()

#    shap.summary_plot(shap_interaction_values, train_df,
#              plot_type="compact_dot", show=False)
#    plt.tight_layout()
#    plt.savefig(os.path.join(
#    dir_res, "{}_shap_interactions.png".format(prefix)))
#    plt.clf()

#    inds = shap.approximate_interactions("mass", shap_values, train_df)
#    # make plots colored by each of the top three possible interacting features
#    for i in range(3):
#    if i >= features_df.shape[1]:
#    continue
#    shap.dependence_plot("mass", shap_values, train_df,
#                     interaction_index=inds[i], show=False)
#    plt.tight_layout()
#    plt.savefig(os.path.join(
#    dir_res, "{}_mass_dependence_and_{}.png".format(prefix, i)))
#    plt.clf()