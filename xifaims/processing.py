"""
Created on Wed Oct 14 21:55:31 2020

@author: hanjo
"""
import re
import numpy as np
import pickle
import pandas as pd
import os
from xifaims import features as xf


def process_csms(infile_loc, config):
    """
    Read inputfile, remove non-unique ids and split data
    """
    # read file and annotate CV
    df = pd.read_csv(infile_loc)
    # set cv
    df["CV"] = - df["run"].apply(get_faims_cv)
    df = df.drop(df.filter(regex="Unnamed").columns, axis=1)
    df = df.set_index("PSMID")
    # filter by charge
    # df_unique, df_nonunique =
    return preprocess_nonunique(df)


def process_features(df_TT, one_hot, config):
    """
    Process the splitted dataframes for TTs and DDS and add features for both dataframes.

    df_TT: df, TTs
    df_DX: df, DXs
    one_hot: bool, if one hot encoding for the charge should be used
    config: dic, config dict, used for including / excluding features
    """
    tmp_config = {i: j for i, j in config.items()}
    # compute features
    # here it is possible to adjust the charge coding either as one-hot or continous feature
    # drop_features = ["proline", "DE", "KR", "log10mass", "Glycine"]
    df_features = xf.compute_features(df_TT, onehot=one_hot).drop(tmp_config["exclude"], axis=1)

    # only filter if include is specified, else just take all columns
    if not one_hot:
        charges = [f"charge_{d}" for d in np.arange(2, 9)]
        tmp_config["include"] = [i for i in config["include"] if i not in charges]
        tmp_config["include"].append("p.charge")

    # if whielists are used to include features, filter the feature df here
    if len(tmp_config["include"]) > 1:
        df_features = df_features[tmp_config["include"]]
    return df_features


def get_faims_cv(run, acq="LS"):
    """
    Annotate data with faims CV.

    Parameters
    ----------
    run : str
        run name.

    Returns
    -------
    float
        cv extract from run.

    """
    if acq == "LS":
        try:
            return float(re.search(r"CV(\d+)", run).groups()[0])
        except AttributeError:
            # mixed data also possible ...
            return float(re.search(r"_(\d+)_", run).groups()[0])
    else:
        return float(re.search(r"_(\d+)_", run).groups()[0])


def preprocess_nonunique(df_psms):
    """
    Return unique dataframe.

    Parameters
    ----------
    df1 : TYPE
        DESCRIPTION.

    Returns
    -------
    df2 : TYPE
        DESCRIPTION.

    """
    # columsn to create unique id
    # grp_cols = ["PepSeq1", "PepSeq2", "exp charge", "LinkPos1", "LinkPos2"]
    grp_cols = ["PepSeq1", "PepSeq2", "exp charge"]
    # sort by fdr to keep best
    df_psms = df_psms.sort_values(["fdr"])
    print("input psms: ", df_psms.shape)
    # set a dictionary that defines the used function (iloc[0]) for the groups
    # can be better solved by just dropping the other columns actually ...
    # param = {i: lambda x: x.iloc[0] for i in df_psms.columns[0:-1] if i not in grp_cols}
    # param["CV"] = "mean"

    # df_psms_unique = df_psms.groupby(grp_cols, as_index=False).agg(param)
    # print("aggregated psms: ", df_psms_unique.shape)
    df_psms["mCV"] = df_psms.groupby(grp_cols)['CV'].transform('mean')
    df_psms_unique = df_psms.drop_duplicates(grp_cols, keep="first")
    print("unique: ", df_psms_unique.shape)
    return df_psms_unique, df_psms


def split_target_decoys(df_psms, frac=1, random_state=42):
    """
    Shuffle and return TTs only.

    Parameters
    ----------
    df_psms : TYPE
        DESCRIPTION.
    frac : TYPE, optional
        DESCRIPTION. The default is 1.
    random_state : TYPE, optional
        DESCRIPTION. The default is 42.

    Returns
    -------
    None.

    """
    df_psms = df_psms.sample(frac=frac, random_state=random_state)
    df_TT = df_psms[df_psms.isTT]
    df_TT = df_TT.reset_index(drop=True)

    df_DX = df_psms[~df_psms.isTT]
    df_DX = df_DX.reset_index(drop=True)
    return df_TT, df_DX

def store_for_shap(df_TT, df_TT_features, df_DX, df_DX_features, classifier, path="", prefix=""):
    all_objs = {"TT": df_TT,
                "TT_feat": df_TT_features,
                "DX": df_DX,
                "DX_features": df_DX_features,
                "clf": classifier}

    pickle.dump(all_objs, open(os.path.join(path, f"{prefix}_shap_data.p"), "wb"))


def load_for_shap(path="", prefix=""):
    shap_data = pickle.load(open(os.path.join(path, f"{prefix}_shap_data.p"), "rb"))
    return shap_data["TT"], shap_data["TT_feat"], shap_data["DX"], shap_data["DX_features"], \
           shap_data["clf"]


def charge_filter(df, charge):
    """
    Return a charge-filtered dataframe.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    charge : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if charge == "all":
        return df

    elif charge == 3:
        return df[df["exp charge"] == 3]

    elif charge == 4:
        return df[df["exp charge"] == 4]

    elif charge == 5:
        return df[df["exp charge"] == 5]


def find_nearest(value, array):
    """Return closes value in the array that matches the input value."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
