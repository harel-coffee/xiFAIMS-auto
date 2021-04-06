# xiFAIMS
Module for analysis of crosslinking MS/FAIMS data.

To reproduce the analysis in the manuscript install the requirements from the pipefile and run:

>snakemake -s .\run_xifaims.sn -j 8 --printshellcmds 

Then open the **xifaims_xgb_notebook** to generate the ML results.


# dev notes

## Modules
List of available modules:
- const - stores amino acids constants for feature computation
- features - compute and manage feature computation
- ml - perform, document and store machine learning results
- parameters - store XGB parameter grids
- plots - visual presentation of results
- processing - various pre and post processing tools
- seq_lib - processing functions for sequences borrowed from xiRT

## methods summary
- 80% train, 20% test
- with 80% -> 3-fold cross-validation
- minimize neg_mean_squared_error in sklearns gridsearch


## xifaims_xgb overview

The main script that is executed via snakemake is **xifaims_xgb.py**. This file does the following

**goal:** build a predictor (xgboost) for CV based on sequence features.

**steps:**
- parse prepared dataframe with csms
- only use unique peptides (alpha/beta peptide and charge)
- only use TT for machine learning
- splits the data into 80/20 (training / validation)
- compute sequence-based features
- perform hyper parameter optimization for xgboost regressor on the 80% split
- if enabled perform feature selection and extract most predictive features
- compute metrics from training / validation split
- store meta data / data in a pickle file and excel file (for all possible objects)

## parameters

The main script can be parameterized to only use specific sets of features, hyper parameters.
The parameters folder has a couple of examples (e.g. faims_all.yaml). Further documentation on
the command line arguments can be retrieved by executing --help on the terminal.

## post optimization analysis

Open the **xifaims_xgb_notebook** to "interactively" go through the results.

# installation
Clone this repo and then install via pip (pip install -e .). Make sure to install the dependencies,
xirt etc.

# Contributors
- Sven Giese
- Ludwig Sinn