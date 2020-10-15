# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:18:36 2020

@author: hanjo
"""

# import pandas as pd
# df = pd.read_excel("data_new/features_LS.xlsx")
# print (dict(zip(df["Amino acid"], df["Polarity"])))
# print (dict(zip(df["Amino acid"], df["Secondary_structure"])))
# print (dict(zip(df["Amino acid"], df["Molecular_volume"])))
# print (dict(zip(df["Amino acid"], df["Electrostatic_charge"])))
# print (dict(zip(df["Amino acid"], df["Intrinsic_size_B"])))


polarity_map = {'A': -0.591,
                'C': -1.343,
                'D': 1.05,
                'E': 1.357,
                'F': -1.006,
                'G': -0.384,
                'H': 0.336,
                'I': -1.239,
                'K': 1.831,
                'L': -1.019,
                'M': -0.663,
                'N': 0.945,
                'P': 0.189,
                'Q': 0.931,
                'R': 1.538,
                'S': -0.228,
                'T': -0.032,
                'V': -1.337,
                'W': -0.595,
                'Y': 0.26}

secondarystructure_map = {'A': -1.302, 'C': 0.465, 'D': 0.302, 'E': -1.453, 'F': -0.59, 'G': 1.652,
                          'H': -0.417, 'I': -0.547, 'K': -0.561,
                          'L': -0.987, 'M': -1.524, 'N': 0.828, 'P': 2.081, 'Q': -0.179,
                          'R': -0.055, 'S': 1.399, 'T': 0.326, 'V': -0.279, 'W': 0.009, 'Y': 0.83}

molecularvolumne_map = {'A': -0.733, 'C': -0.862, 'D': -3.656, 'E': 1.477, 'F': 1.891,
                        'G': 1.33, 'H': -1.673, 'I': 2.131, 'K': 0.533,
                        'L': -1.505, 'M': 2.219, 'N': 1.299, 'P': -1.628, 'Q': -3.005,
                        'R': 1.502, 'S': -4.76, 'T': 2.213, 'V': -0.544, 'W': 0.672, 'Y': 3.097}

electrostaticcharge_map = {'A': -0.146, 'C': -0.255, 'D': -3.242, 'E': -0.837, 'F': 0.412,
                           'G': 2.064, 'H': -0.078, 'I': 0.816, 'K': 1.648,
                           'L': -0.912, 'M': 1.212, 'N': 0.933, 'P': -1.392, 'Q': -1.853,
                           'R': 2.897, 'S': -2.647, 'T': 1.313, 'V': -1.262, 'W': -0.184,
                           'Y': 1.512}

intrnsicsize_map = {'A': 1.004, 'C': 0.865, 'D': 0.883, 'E': 0.919, 'F': 1.026, 'G': 0.899,
                    'H': 1.165, 'I': 1.09, 'K': 1.23,
                    'L': 1.13, 'M': 0.977, 'N': 0.879, 'P': 0.918, 'Q': 0.918, 'R': 1.15,
                    'S': 0.899, 'T': 0.946, 'V': 1.05, 'W': 0.941, 'Y': 1.01}
