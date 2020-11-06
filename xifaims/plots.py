"""
Created on Wed Oct 14 22:50:36 2020

@author: hanjo
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statannot import add_stat_annotation


def const_line(*args, **kwargs):
    cvs = np.array([30., 35., 40., 45., 50., 55.,
                    60., 65., 70., 75., 80., 85., 90.])
    cvs = cvs * (-1)
    plt.plot(cvs, cvs, c='k')
    
    
def feature_correlation_plot(features_df, outpath, prefix="", show=False):
    """
    Plot a correlation matrix from the input features.

    Parameters
    ----------
    features_df : TYPE
        DESCRIPTION.
    outpath : TYPE
        DESCRIPTION.
    show : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    # correlation
    cor = features_df.corr().round(1).abs()
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    # relevant_features = cor[cor > 0.5]
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(outpath, prefix + "correlation_matrix_features.pdf"))
        plt.clf()


def train_test_scatter_plot(all_clf, outpath, prefix="", show=False):
    """PLot a observed vs. pedicted scatter plot."""
    g = sns.FacetGrid(all_clf, col="Set", row="classifier")
    g = g.map(plt.scatter, "CV_Train", "CV_Predict")
    g = g.map(const_line)
    g.set(ylim=(-90, -20))
    g.set(xlim=(-90, -20))
    g.set_xlabels("Observed CV")
    g.set_ylabels("Predicted CV")
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(outpath, prefix + "train_test_scatter.pdf"))
        plt.clf()


def cv_performance_plot(all_metrics, outpath, prefix="", show=False):
    """Plot a barplot as perfomrance overview."""
    # second plot
    all_metrics["abs_mse"] = all_metrics["mse"].abs()
    sns.barplot(x="classifier", y="abs_mse", hue="split", data=all_metrics,
                hue_order=["train", "test"])
    sns.despine()
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(outpath, prefix + "train_test_scatter.pdf"))
        plt.clf()


def plot_search_results(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    # Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    # Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data == p_v))

    params = grid.param_grid

    # Ploting results
    fig, ax = plt.subplots(1, len(params),
                           sharex='none', sharey='all', figsize=(20, 5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        try:
            x = np.array(params[p])
        except:
            x = np.array(params[0][p])
        x = np.array([i if i != None else -1 for i in x])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        try:
            ax[i].errorbar(x, y_1, e_1, linestyle='--',
                           marker='o', label='test')
            ax[i].errorbar(x, y_2, e_2, linestyle='-',
                           marker='^', label='train')
            ax[i].set_xlabel(p.upper())
        except:
            ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
            ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^', label='train')
            ax[i].set_xlabel(p.upper())
    plt.legend()


def target_decoy_comparison(prediction_df):
    """
    Plot the prediction differences between TTs and DXs.
    """
    prediction_df["error"] = prediction_df["Observed CV"] - prediction_df["Predicted CV"]
    f, ax = plt.subplots(1, figsize=(3, 4))
    if len(prediction_df["Type"].drop_duplicates()) == 3:
        order = ["TT", "TD", "DD"]
    else:
        order = ["TT", "TD"]
    sns.boxenplot(data=prediction_df, y="error", x="Type", ax=ax, order=order)
    ax.axhline(0.0, lw=2, c="k", alpha=0.7, zorder=-1)
    ax.set(ylabel="CV prediction error")
    sns.despine(ax=ax)
    test_results = add_stat_annotation(ax, data=prediction_df, x="Type", y="error", order=order,
                                       box_pairs=[("TT", "TD")],
                                       test='t-test_ind', text_format='star',
                                       loc='outside', verbose=2)
    plt.savefig("notebooks/TT_TD_prediction_error_box.png")
    plt.savefig("notebooks/TT_TD_prediction_error_box.svg")
    plt.show()

    f, ax = plt.subplots(1, figsize=(3, 4))
    sns.histplot(data=prediction_df, x="error", hue="Type", stat="density", element="step",
                 common_norm=True, ax=ax)
    ax.axhline(0.0, lw=2, c="k", alpha=0.7, zorder=-1)
    ax.set(ylabel="CV prediction error")
    sns.despine(ax=ax)
    plt.savefig("notebooks/TT_TD_prediction_error_hist.png")
    plt.savefig("notebooks/TT_TD_prediction_error_hist.svg")
    plt.show()
