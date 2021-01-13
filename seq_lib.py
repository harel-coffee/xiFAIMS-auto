"""Module to process peptide sequences."""
import re
from collections import Counter
import numpy as np
from pyteomics import parser
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import sequence as ts
import pandas as pd


def prepare_seqs(psms_df, seq_cols=["Peptide1", "Peptide2"]):
    """Convert Peptide sequences to unified format.
    This conversion simplifies the alphabet of the amino acids, removes special characters,
    replaces digits to written numbers, removes n-terminal modifications, rewrites Xmod
    modifications to modX format and splits the amino acid string into a list.
    Args:
        psms_df:  df, psm dataframe
        seq_cols: list, list of columns names that contain the peptide sequences with
        modifications
    Returns:
        df, sequence arrays
    Information:
        Performs the following steps:
        - simplifie amino acid alphabet by replacing non-sandard aa
        - remove special characters (brackets, _)
        - replace numbers by written numbers (3 -> three)
        - remove n-terminal mods since they are currently not supported
        - rewrite peptide sequence to modX format (e.g. Mox -> oxM)
        - parse a string sequence into a list of amino acids, adding termini
    """
    # for all sequence columns in the dataframe perform the processing
    # if linear only a single column, if cross-linked two columns are processed
    for seq_col in seq_cols:
        # code if sequences are represented with N.SEQUENCE.C
        if "." in psms_df.iloc[0][seq_col]:
            psms_df["Seq_" + seq_col] = psms_df[seq_col].str.split(".").str[1]
        else:
            psms_df["Seq_" + seq_col] = psms_df[seq_col]

        sequences = psms_df["Seq_" + seq_col]
        sequences = sequences.apply(simplify_alphabet)
        sequences = sequences.apply(remove_brackets_underscores)
        sequences = sequences.apply(replace_numbers)
        sequences = sequences.apply(remove_nterm_mod)
        sequences = sequences.apply(rewrite_modsequences)
        sequences = sequences.apply(parser.parse, show_unmodified_termini=True)
        psms_df["Seqar_" + seq_col] = sequences
    return(psms_df)


def simplify_alphabet(sequence):
    """Replace ambiguous amino acids.

    Some sequences are encoded with 'U', arbitrarily choose C as residue to
    replace any U (Selenocystein).

    Parameters:
    ----------
    sequence: string,
              peptide sequences
    """
    return sequence.replace("U", "C")


def remove_brackets_underscores(sequence):
    """Remove all brackets (and underscores...) from protein sequences.

    Needed for MaxQuant processing.

    Parameters:
    ----------
    sequence: string,
              peptide sequences
    """
    return re.sub("[\(\)\[\]_\-]", "", sequence)


def replace_numbers(sequence):
    """Replace digits to words (necessary for modX format to work.

    :param sequence:
    :return:
    """
    rep = {"1": "one",
           "2": "two",
           "3": "three",
           "4": "four",
           "5": "five",
           "6": "six",
           "7": "seven",
           "8": "eight",
           "9": "nine",
           "0": "zero"}
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], sequence)


def remove_nterm_mod(sequence):
    """Remove the nterminal modification.

    Meant to be used for "ac" modifications in front of the sequence.
    They are not currently supported and need to be removed.

    :param sequence: str, peptide sequence
    :return:
    """
    return re.sub(r'^([a-z]+)([A-Z])', r'\2', sequence, flags=re.MULTILINE)


def rewrite_modsequences(sequence):
    """Rewrite modified sequences to modX format.

    Requires the input to be preprocessed such that no brackets are in the sequences.

    Meant to be used via apply.

    Example:
    -------
    sequence = "ELVIS"
    sequence = "ELVISCcmASD"

    :param sequence: str, peptide sequence
    :return:
    """
    return re.sub("([A-Z])([^A-Z]+)", r'\2\1', sequence)


def remove_lower_letters(sequence):
    """Remove lower capital letters from the sequence.

    :param sequence: str, peptide sequence
    :return:
    """
    return re.sub("[a-z]", "", sequence)


def to_unmodified_sequence(sequence):
    """Remove lower capital letters, brackets from the sequence.

    :param sequence: str, peptide sequence
    :return:
    """
    return (re.sub("[^[A-Z]", "", sequence))


def reorder_sequences(matches_df):
    """Reorder peptide sequences by length.

    Defining the longer peptide as alpha peptide and the shorter petpide as
    beta peptide. Ties are resolved lexicographically.

    Args:
        matches_df: df, dataframe with peptide identifications.

    Returns:
        df, dataframe with the swapped cells and additional indicator column ('swapped')
    """
    # compile regex to match columsn with 1/2 in the end of the string
    # check if all pairwise columns are there
    r = re.compile(r"\w+(?:1$|2$)")
    # get matching columns
    match_columns = list(filter(r.match, matches_df.columns))
    # remove number
    pairs_noidx = [i[:-1] for i in match_columns]
    # count occurrences and check if pairs are there
    counts = [(i, j) for i, j in Counter(pairs_noidx).items() if j == 2]
    if len(counts) * 2 != len(pairs_noidx):
        raise ValueError("Error! Automatic column matching could not find pairwise crosslink "
                         "columns. Please make sure that you have a peptide1, peptide2 "
                         "column name pattern for crosslinks. Columns must appear in pairs if there"
                         " is a number in the end of the name.")

    # order logic, last comparison checks lexigographically
    is_longer = (matches_df["Peptide1"].apply(len) > matches_df["Peptide2"].apply(len)).values
    is_shorter = (matches_df["Peptide1"].apply(len) < matches_df["Peptide2"].apply(len)).values
    is_greater = (matches_df["Peptide1"] > matches_df["Peptide2"]).values

    # create a copy of the dataframe
    swapping_df = matches_df.copy()
    swapped = np.ones(len(matches_df), dtype=bool)

    # z_idx for 0-based index
    # df_idx for pandas
    for z_idx, df_idx in enumerate(matches_df.index):
        if not is_shorter[z_idx] and is_greater[z_idx] or is_longer[z_idx]:
            # for example: AC - AA, higher first, no swapping required
            swapped[z_idx] = False
        else:
            # for example: AA > AC, other case, swap
            swapped[z_idx] = True

        if swapped[z_idx]:
            for col in pairs_noidx:
                swapping_df.at[df_idx, col + str(2)] = matches_df.iloc[z_idx][col + str(1)]
                swapping_df.at[df_idx, col + str(1)] = matches_df.iloc[z_idx][col + str(2)]
    swapping_df["swapped"] = swapped
    return swapping_df


def modify_cl_residues(matches_df, seq_in=["Peptide1", "Peptide2"]):
    """
    Change the cross-linked residues to modified residues.

    Function uses the Seqar_*suf columns to compute the new peptides.

    Args:
        matches_df: df, dataframe with peptide identifications. Required columns
        seq_in:

    Returns:
        psms_df: df, dataframe with adapted sequences in-place
    """
    # increase the alphabet by distinguishing between crosslinked K and non-crosslinked K
    # introduce a new prefix cl for each crosslinked residue
    for seq_id, seq_i in enumerate(seq_in):
        for idx, row in matches_df.iterrows():
            residue = row["Seqar_" + seq_i][row["LinkPos" + str(seq_id + 1)]]
            matches_df.at[idx, "Seqar_" + seq_i][row["LinkPos" + str(seq_id + 1)]] = "cl" + residue


def get_mods(sequences):
    """Retrieve modifciations from dataframe in the alphabet.

    Parameters
    ----------
    sequences : ar-like
        peptide sequences.

    Returns
    -------
    List with modification strings.
    """
    return np.unique(re.findall("-OH|H-|[a-z0-9]+[A-Z]", " ".join(sequences)))


def get_alphabet(sequences):
    """Retrieve alphabet of amino acids with modifications.

    Parameters
    ----------
    sequences : ar-like
        peptide sequences.

    Returns
    -------
    List with modification strings.
    """
    return np.unique(
        re.findall("-OH|H-|[a-z0-9]+[A-Z]|[A-Z]", " ".join(sequences))
    )


def label_encoding(sequences, max_sequence_length, alphabet=[], le=None):
    """Label encode a list of peptide sequences.

    Parameters
    ----------
    sequences : ar-like
        list of amino acid characters (n/c-term/modifications.
    max_sequence_length : int
        maximal sequence length (for padding).
    alphabet : list, optional
        list of the unique characters given in the sequences
    le : TYPE, optional
        label encoder instance, can be a prefitted model.

    Returns
    -------
    X_encoded : TYPE
        DESCRIPTION.
    """
    # if no alternative alphabet is given, use the defaults
    if len(alphabet) == 0:
        alphabet = parser.std_amino_acids

    if not le:
        # init encoder for the AA alphabet
        le = LabelEncoder()
        le.fit(alphabet)

    # use an offset of +1 since shorter sequences will be padded with zeros
    # to achieve equal sequence lengths
    X_encoded = sequences.apply(le.transform) + 1
    X_encoded = ts.pad_sequences(X_encoded, maxlen=max_sequence_length)
    return X_encoded, le


def generate_padded_df(encoded_ar, index):
    """Generate indexed dataframe from already label-encoded sequences.
    Args:
        encoded_ar:
        index:
    Returns:
        dataframe, label encoded peptides as rows
    """
    seqs_padded = pd.DataFrame(encoded_ar, index=index)
    seqs_padded.columns = ["rnn_{}".format(str(i).zfill(2)) for i in np.arange(len(encoded_ar[0]))]
    return seqs_padded


def featurize_sequences(psms_df, seq_cols=["Seqar_Peptide1", "Seqar_Peptide2"], max_length=-1):
    """Generate a featureized version of sequences from a data frame.
    The featureization is done via by retrieving all modifications, the relevant alphabet of amino
    acids and then applying label encoding do the amino acid sequences.
    :param psms_df: df, dataframe with identifications
    :param seq_cols: list, list with column names
    :param max_legnth: int, maximal length for peptides to be included as feature
    :return:
    """
    # transform a list of amino acids to a feature dataframe
    # if two sequence columns (crosslinks)

    # define lambda function to get the alphabet in both cases
    f = lambda x: get_alphabet(x)

    # get padding length
    if max_length == -1:
        max_length = psms_df[seq_cols].applymap(len).max().max()

    # get amino acid alphabet
    if len(seq_cols) > 1:
        alphabet = np.union1d(f(psms_df[seq_cols[0]].str.join(sep="").drop_duplicates()),
                              f(psms_df[seq_cols[1]].str.join(sep="").drop_duplicates()))
    else:
        alphabet = f(psms_df[seq_cols[0]].str.join(sep="").drop_duplicates())

    # perform the label encoding + padding
    encoded_s1, le = label_encoding(psms_df[seq_cols[0]], max_length, alphabet=alphabet)
    seq1_padded = generate_padded_df(encoded_s1, psms_df.index)

    if len(seq_cols) > 1:
        encoded_s2, _ = label_encoding(psms_df[seq_cols[1]], max_length, alphabet=alphabet,
                                          le=le)
        seq2_padded = generate_padded_df(encoded_s2, psms_df.index)
    else:
        seq2_padded = pd.DataFrame()

    return(seq1_padded, seq2_padded, le)