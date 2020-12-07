"""Module for feature computation.

Created on Wed Oct 14 22:13:28 2020

@author: hanjo
"""
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from xifaims import const
from xirt import features as xirtf
from modlamp.descriptors import GlobalDescriptor
from pyteomics import electrochem


def get_electrochem_charge(peptide, pH=2.8):
    """Compute charge using pyteomics."""
    return electrochem.charge(peptide, pH=pH)


def only_upper(seq):
    """
    Return only upper sequences.

    Parameters
    ----------
    seq : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return "".join([i for i in seq if i.isupper()])


def get_structure_perc(seq, structure="helix"):
    """Use biopython to compute helix, turn, sheet % from the amino acid composition."""
    bio_seq = ProteinAnalysis(seq)
    helix, turn, sheets = bio_seq.secondary_structure_fraction()

    if structure == "helix":
        return helix

    elif structure == "turn":
        return turn

    else:
        return sheets


def compute_features(df, onehot=True):
    """
    Compute features for peptide sequences.

    Parameters
    ----------
    df : df
        df with psms.

    Returns
    -------
    df_features : df
        feature df.

    """
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5750047/

    df["seq1seq2"] = df["PepSeq1"] + df["PepSeq2"]

    # computes standard features
    df_features = pd.DataFrame(index=df.index)
    df_features["length1"] = df["PeptideLength1"]
    df_features["length2"] = df["PeptideLength2"]
    df_features["length1+length2"] = df["PeptideLength1"] + \
        df["PeptideLength2"]
    df_features["mass"] = df["exp mass"]
    df_features["log10mass"] = np.log10(df["exp mass"])

    # modifications
    df_features["loop"] = df["seq1seq2"].str.contains("loop")
    df_features["oh"] = df["seq1seq2"].str.contains("oh")
    df_features["nh2"] = df["seq1seq2"].str.contains("nh2")

    if onehot:
        ohc = pd.get_dummies(df["exp charge"])
        ohc.columns = ["charge_" + str(i) for i in ohc.columns]
        df_features = df_features.join(ohc)
    else:
        df_features["p.charge"] = df["exp charge"]

    df_features["DE"] = df["seq1seq2"].str.count("D") + df["seq1seq2"].str.count("E")
    df_features["KR"] = df["seq1seq2"].str.count("K") + df["seq1seq2"].str.count("R")
    df_features["aromatics"] = df["seq1seq2"].str.count("F") + \
        df["seq1seq2"].str.count("W") + df["seq1seq2"].str.count("Y")

    df_features["proline"] = df["seq1seq2"].str.count("P")
    df_features["Glycine"] = df["seq1seq2"].str.count("G")

    # charge
    #df_features["charge_cmp"] = df["seq1seq2"].apply(get_electrochem_charge)
    
    # biopython
    df_features["helix"] = df["seq1seq2"].apply(get_structure_perc, args=("helix",))
    df_features["sheet"] = df["seq1seq2"].apply(get_structure_perc, args=("sheet",))
    df_features["turn"] = df["seq1seq2"].apply(get_structure_perc, args=("turn",))
    df_features["pi"] = df["seq1seq2"].apply(xirtf.get_pi)

    # modlamp features
    seqs = [only_upper(i) for i in df["seq1seq2"]]
    glob = GlobalDescriptor(seqs)

    glob.calculate_charge(ph=2.8, amide=True)
    df_features["charge_glob"] = np.ravel(glob.descriptor)

    glob.charge_density(ph=2.8, amide=True)
    df_features["charge_density"] = np.ravel(glob.descriptor)

    glob.hydrophobic_ratio()
    df_features["hydrophobic_ratio"] = np.ravel(glob.descriptor)

    df_features["intrinsic_size_sum"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.intrnsicsize_map, np.sum))
    df_features["intrinsic_size_std"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.intrnsicsize_map, np.std))
    df_features["intrinsic_size_max"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.intrnsicsize_map, np.max))

    df_features["mv_sum"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.molecularvolumne_map, np.sum))
    df_features["mv_size_std"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.molecularvolumne_map, np.std))
    df_features["mv_size_max"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.molecularvolumne_map, np.max))

    df_features["polarity_sum"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.polarity_map, np.sum))
    df_features["polarity_size_std"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.polarity_map, np.std))
    df_features["polarity_size_max"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.polarity_map, np.max))

    df_features["secondstruc_sum"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.secondarystructure_map, np.sum))
    df_features["secondstruc_size_std"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.secondarystructure_map, np.std))
    df_features["secondstruc_size_max"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.secondarystructure_map, np.max))

    df_features["estatic_sum"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.electrostaticcharge_map, np.sum))
    df_features["sstatic_size_std"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.electrostaticcharge_map, np.std))
    df_features["static_size_max"] = df["seq1seq2"].apply(
        fsummarizer, args=(const.electrostaticcharge_map, np.max))

    df_features["m/z"] = df["exp m/z"]

    return df_features


def fsummarizer(pepseq, prop_dict, func):
    """
    Summarize pepide sequence with by mapping amino acids to prop_dict entires and a summary func.

    Parameters
    ----------
    pepseq : str
        sequence.
    prop_dict : dict
        amino acid property mapping.
    func : function
        Summary function (sum, std, max, ..).

    Returns
    -------
    func applied to mapped AA outcomes.

    """
    return func([prop_dict[i] for i in pepseq if i.isupper()])
