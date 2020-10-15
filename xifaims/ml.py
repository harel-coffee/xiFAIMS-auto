"""
Created on Wed Oct 14 22:57:36 2020

@author: hanjo

!autopep8 xifaims/ml.py -i
!isort xifaims/ml.py
!autoflake --in-place --remove-unused-variables xifaims/ml.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
# from xgboost_autotune import fit_parameters


def training(df_TT, df_TT_features, model="SVM", scale=True):
    """
    Perform training a with variable models on the data.

    Parameters
    ----------
    df_TT : TYPE
        DESCRIPTION.
    df_TT_features : TYPE
        DESCRIPTION.
    model : TYPE, optional
        DESCRIPTION. The default is "SVM".
    scale : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # train, validation split
    training_idx, validation_idx = train_test_split(df_TT.index, test_size=0.1)

    # global scaler
    ss_train = MinMaxScaler().fit(df_TT_features.loc[training_idx])

    # x variable
    if scale:
        val_df = ss_train.transform(df_TT_features.loc[validation_idx])
        train_df = ss_train.transform(df_TT_features.loc[training_idx])

        train_df = pd.DataFrame(
            train_df, index=training_idx, columns=df_TT_features.columns)
        val_df = pd.DataFrame(val_df, index=validation_idx,
                              columns=df_TT_features.columns)
    else:
        val_df = df_TT_features.loc[validation_idx]
        train_df = df_TT_features.loc[training_idx]

    # y variable
    train_y = df_TT["CV"].loc[training_idx]
    val_y = df_TT["CV"].loc[validation_idx]

    # fit the baseline
    if model == "SVM":
        return SVM_baseline(train_df, train_y, val_df, val_y)

    elif model == "XGB":
        return XGB_model(train_df, train_y, val_df, val_y)


def SVM_baseline(train_df, train_y, val_df, val_y, cv=3):
    """
    Fits a SVM baseline model with exhaustive parameter optimization.

    Parameters
    ----------
    train_df : TYPE
        DESCRIPTION.
    train_y : TYPE
        DESCRIPTION.
    val_df : TYPE
        DESCRIPTION.
    val_y : TYPE
        DESCRIPTION.

    Returns
    -------
    df_results : TYPE
        DESCRIPTION.
    cv_res : TYPE
        DESCRIPTION.

    """
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # grid search
    gs = GridSearchCV(SVR(), tuned_parameters, cv=cv, scoring="neg_mean_squared_error",
                      return_train_score=True, verbose=1, n_jobs=8)
    gs.fit(train_df, train_y)

    # refit clf
    svc = SVR(**gs.best_params_)
    svc.fit(train_df, train_y)

    # get summary results
    df_results, cv_res = format_summary(
        train_df, val_df, train_y, val_y, svc, "SVM", gs)
    cv_res["params"] = str(gs.best_params_)
    return df_results, cv_res, gs, svc


def format_summary(train_df, val_df, train_y, val_y, clf, clf_name, gs):
    """
    Create summary dataframes for classifier performance.

    Parameters
    ----------
    train_df : TYPE
        DESCRIPTION.
    val_df : TYPE
        DESCRIPTION.
    train_y : TYPE
        DESCRIPTION.
    val_y : TYPE
        DESCRIPTION.
    clf : TYPE
        DESCRIPTION.
    clf_name : TYPE
        DESCRIPTION.
    gs : TYPE
        DESCRIPTION.

    Returns
    -------
    df_results : TYPE
        DESCRIPTION.
    cv_res : TYPE
        DESCRIPTION.

    """
    # make predictions
    df_results = pd.DataFrame()
    df_results["CV_Train"] = np.concatenate([train_y, val_y])
    df_results["CV_Predict"] = np.concatenate(
        [clf.predict(train_df), clf.predict(val_df)])
    df_results["Set"] = np.concatenate(
        [["Train"] * len(train_y), ["Test"] * len(val_y)])
    df_results["classifier"] = clf_name

    # format results
    cv_res = pd.DataFrame(gs.cv_results_)
    cv_res = cv_res.sort_values(by="rank_test_score", ascending=True).iloc[0]
    cv_res = pd.DataFrame(cv_res.filter(regex="split")).reset_index()
    cv_res.columns = ["cv_split", "mse"]
    cv_res["split"] = cv_res["cv_split"].str.split("_").str[1]
    cv_res["classifier"] = clf_name
    return df_results, cv_res


def FAIMSNETNN_model(input_dim=10):
    """FIT neuralnetwork model."""
    inputs = keras.Input(shape=(input_dim,))
    x = Dense(1000, activation="relu", kernel_regularizer="l2")(inputs)
    x = Dropout(0.15)(x)

    x = Dense(500, activation="relu", kernel_regularizer="l2")(inputs)
    x = Dropout(0.1)(x)

    x = Dense(250, activation="relu", kernel_regularizer="l2")(x)
    x = Dropout(0.15)(x)

    x = Dense(100, activation="relu", kernel_regularizer="l2")(x)
    x = Dropout(0.15)(x)

    outputs = Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="FAIMSnet")
    model.summary()
    return model


def mle_feature_selection():
    # feature selection
    xgbr = xgboost.XGBRFRegressor(**xgbr1.get_params())
    
    sfs = SFS(xgbr, k_features=8, forward=True,
              floating=False, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    sfs = sfs.fit(train_df, train_y)
    
    
    features = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    print(features)
    
    fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.grid()
    plt.show()


def XGB_model(train_df, train_y, val_df, val_y):
    """
    Fits XGB model with exhaustive parameter optimization.

    Parameters
    ----------
    train_df : TYPE
        DESCRIPTION.
    train_y : TYPE
        DESCRIPTION.
    val_df : TYPE
        DESCRIPTION.
    val_y : TYPE
        DESCRIPTION.

    Returns
    -------
    df_results : TYPE
        DESCRIPTION.
    cv_res : TYPE
        DESCRIPTION.

    """
    # %% round 1
    # XGBoost
    # parameters = {
    #     'n_estimators': [50, 150],
    #     'max_depth': [3, 6, 9],
    #     'learning_rate': [0.1, 0.05],
    #     'subsample': [0.9, 1.0],
    #     'reg_alpha': [1.1, None],
    #     'colsample_bytree': [0.5, 0.7],
    #     'min_child_weight': [1, 3],
    #     'gamma': [0.0, 0.1],
    #     'nthread': [8],
    #     'seed': [42]}

    # xgbr = xgboost.XGBRegressor()
    # gs = GridSearchCV(estimator=xgbr, param_grid=parameters, cv=3, n_jobs=1,
    #                   verbose=1, scoring="neg_mean_squared_error",
    #                   return_train_score=True)
    # gs.fit(train_df, train_y)

    # # plot results
    # flib.plot_search_results(gs)
    # plt.savefig("grid_results1.png")
    # # save best results
    # best_params = gs.best_params_
    # print(best_params)
    # %% round 2
    parameters = {'n_estimators': [150, 250, 25],
                  'max_depth': [6, 9, 12],
                  'learning_rate': [0.1, 0.3],
                  'subsample': [0.90],
                  'reg_alpha': [None],
                  'colsample_bytree': [0.7],
                  'min_child_weight': [1, 3],
                  'gamma': [0.0],
                  'nthread': [8],
                  'seed': [42]}

    # xgb model and grid search
    xgbr = xgboost.XGBRegressor()
    gs = GridSearchCV(estimator=xgbr, param_grid=parameters, cv=3, n_jobs=1,
                      verbose=1, scoring="neg_mean_squared_error", return_train_score=True)
    gs.fit(train_df, train_y)

    # refit clf
    xgbr = xgboost.XGBRegressor(**gs.best_params_)
    xgbr.fit(train_df, train_y)

    df_results, cv_res = format_summary(
        train_df, val_df, train_y, val_y, xgbr, "XGB", gs)
    cv_res["params"] = str(gs.best_params_)
    return df_results, cv_res, gs, xgbr
