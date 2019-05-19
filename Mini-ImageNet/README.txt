Write a convolutional network in Keras to train the Mini-ImageNet 
dataset. Your constraint is to create a network
that achieves at least 70% test accuracy.

Submit your answers as two files train.py and test.py. Make
train.py take three inputs: the input training data, training labels,
and a model file name to save the model to. 

python train.py <train.npy> <trainlablels.npy> <model file>

Make test.py take three inputs: the input test data, test labels,
and a model file name to load the model. 

python test.py <test.npy> <testlabels.npy> <model file>

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set.
