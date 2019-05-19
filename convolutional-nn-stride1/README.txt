Write a Python program that trains a neural network with a single 2x2
convolutional layer with stride 1 and global average pooling. See
our course notes on google drive for equation updates with sigmoid
activation. 

The input are 3x3 images. Images for training are going to be in
one directory called train and test ones in the directory called
test. The train directory has a csv file called data.csv that contains
the name of each image dataset and its label. For example your data.csv
would look like

Name,Label
image0.txt,1
image1.txt,-1

where image0.txt is 

1 0 0
0 1 0
0 0 1

and image1.txt is 

0 0 1
0 1 0
1 0 0

Let your program command line be:

python convnet.py <train> <test> 

1. What is the convolutional kernel learnt by your program? 
