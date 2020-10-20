# xiFAIMS

Module for analysis of crosslinking MS and FAIMS data.

## Modules
List of available modules:
- const - stores amino acids constants for feature computation
- explainer - stores code to use SHAP for explainable AI
- features - compute and manage feature computation
- ml - perform, document and store machine learning results
- plots - visual presentation of results
- processing - various pre and post processing tools

## installation
Clone this repo and then install via pip (pip install -e .). Make sure to install the dependencies,
xirt etc.

## performing the analysis

The analysis is divided into three parts:
- xirt usage for CV prediction
- "xifaims" analysis with traditional ML algorithms
- shap analysis for the xifaims analysis


## feature selection

There a couple of feature selection steps involved in the xifaims script. We use the YAML files in the
parameters folder (parameters/faims_all.yaml) to define the different feature sets:

- minimal -> minimal set of features
- smaller -> more features
- small -> even more featues
- structure -> add structural biopython features
- all -> all features discussed

### running xirt
To run the various experiments with xiRT, navigate to the bash folder and simply run the
run_xirt.bat. This script will run a combination of parameters for xiRT (regression, ordinal) +-
auxilarry tasks. The results will be stored as tables (summaries) and plots (for each cv).
The resulting table must be analyzed in the notebook xirt_summary.ipynb

### running xifaims analysis
This module performs the "standard" ML approach using xgboost etc. Here different feature 
combinations are used.


### bash scripts

For convenience the *.bat files contain all python calls with the various options.


## results summary
*please see the notebooks in the mean time*
TBD