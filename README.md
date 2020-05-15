# GCNMDA
**GCNMDA: A novel Graph Convolution Network based framework for predicting Microbe-Drug Associations.**

# Data description
* adj: interaction pairs between microbes and drugs.
* drugs/viruses: IDs and names for drugs/viruses.
* microbes: IDs and names for microbes.
* drugfeatures: pre-processing feature matrix for drugs.
* microbefeatures: pre-processing feature matrix for microbes.
* drugsimilarity: integrated drug similarity matrix.
* microbesimilarity: integrated microbe similarity matrix.

# Run steps
1. To generate training data and test data.
2. Run train.py to train the model and obtain the predicted scores for microbe-drug associations.

# Requirements
* GCNMDA is implemented to work under Python 3.7.
* Tensorflow
* numpy
* scipy
* sklearn

