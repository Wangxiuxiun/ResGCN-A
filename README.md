# ResGCN-A
ResGCN-A: Predicting lncRNA-disease associations using Residual Graph Convolution Networks with attention mechanism

## File Description 
attention.py: proposed attention mechanism.  
ModelTrain.py: input data sets, and the model was trained to obtain the potential features of lncRNAs and diseases.  
dataprocessing.py: the code for data processing.  
model.py: the code for ResGCN-A architecture.  
feature_concat.py: get the representations of positive and negative lncRNA-disease pairs.  
clf_train.py: train Extra-Trees classifiers and perform 5-fold cross validation.

## Usage
Enviroment:  
python==3.7  
pytorch==1.5.1  
torch-geometric==1.6.0  
numpy==1.21.6  
pandas==1.3.5  
matplotlib==3.5.3  
scikit-learn==1.0.2  

## Data
origin_lncRNA_disease_associations.csv: the lncRNA-disease associations matrix.  
DGS.csv: the disease gaussian interaction profile kernel similarity.  
DSS.csv: the disease semantic similarity.  
LFS.csv: the lncRNA functional similarity.  
LGS.csv: the lncRNA gaussian interaction profile kernel similarity.  
 
## Run
Model_Train.py-> feature_concat.py-> clf_train.py

##
The source code will be published after the paper is published.
