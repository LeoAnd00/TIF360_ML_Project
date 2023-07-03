# TIF360_ML_Project

Machine learning project in the course TIF360. \
Comparing the abilities of Graph Neural Networks (GNN) and Transformer Neural Networks (TNN) to \
predict quantum mechanical properties of the QM9 dataset. The networks are also compared to a Multilayer Perceptron (MLP) \
using various molecular descriptors from RDKit and Mordred and/or Morgan fingerprints as inputs. These inputs are also added \
to the dense networks of the GNN and TNN to see if performance is improved.  

## Get Started

The code folders contain the code required to train the networks and generate results. utility_functions.py contain some \
defined functions e.g. for scaling or splitting the data.

To be able to train the networks and generate results, you must first calculate descriptors and generate SMILES from the\
xyz-files in the data folder. This is done by running data_pre_processing.npy and ensuring all the options are set to True.
