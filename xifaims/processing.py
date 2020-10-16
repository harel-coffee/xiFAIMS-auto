"""
Created on Wed Oct 14 21:55:31 2020

@author: hanjo
"""
import re


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
        return float(re.search(r"CV(\d+)", run).groups()[0])
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
    df_psms_unique = df_psms.drop_duplicates(grp_cols, keep="first")
    print("unique: ", df_psms_unique.shape)
    return df_psms_unique


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
    df_DX = df_TT.reset_index(drop=True)
    return(df_TT, df_DX)


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
