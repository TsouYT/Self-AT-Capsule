# Self-AT-Capsule
A new Self-attention CapsNet (Self-AT-Capsule) model is proposed and applied to the analysis of the sentiment analysis problem of the paper.

## Code file description
（1）Training and testing code files

 - main-MR.py：Self-AT-Capsule model text classification data training and testing code files (Here is MR as an example)

（2）Model code file

 - network.py：the code of Self-AT-Capsule model and each baseline model 

   -  kimcnn：TextCNN； 
    
   -  capsule_model_A：Capsule-A；
    
   -  capsule_model_B：Capsule-B；
    
   -  capsule_model_at：AT-CapsNet；
    
   -  capsule_model_at_sq：Syntax-AT-CapsNet
    
   -  capsule_model_at_conv3：Self-AT-Capsule

 - atten_utils.py：Attention submodule

 - layer.py：Capsule network layer

 - loss.py：Loss calculation

（3）Data processing code

 - process_data.py：Process single label data

 - process_data _doc.py：Save single label text

（4）data file

 - subj.hdf5: Subj 

 - subj_adj.hdf5：Syntax tree adjacency matrix of Subj

 - rt-polarity.hdf5：MR 

 - rt-polarity_adj.hdf5：Syntax tree adjacency matrix of MR

 - custrev.hdf5：CR

 - custrev_adj.hdf5：Syntax tree adjacency matrix of CR

 - mpqa.hdf5：MPQA

## Experimental environment
 - Python: 3.6.13

 - scikit-learn: 0.24.2

 - tensorflow-gpu: 1.15.0

 - numpy: 1.19.2

 - nltk: 3.4.5

 - Keras: 2.2.4

 - h5py: 3.1.0

## References
Part of code we quote from 

https://github.com/Mr-jxd/Syntax-AT-CapsNet/tree/main

The following URL we quote for datasets

https://github.com/harvardnlp/sent-conv-torch/tree/master/data
