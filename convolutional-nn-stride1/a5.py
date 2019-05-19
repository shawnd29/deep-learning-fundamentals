# This program is used to find the neural network with a single 2x2
#convolutional layer with stride 1 and global average pooling
#To run this program python a5.py <traindirectory> <test directory>
#Done by Shawn D'Souza srd59

import numpy as np
import sys
import os
import pandas as pd
from scipy import signal


###read images from the train directory
traindir = sys.argv[1]
df = pd.read_csv(traindir+'/data.csv')#load images' names and labels
names = df['Name'].values
labels = df['Label'].values 

traindata = np.empty((len(labels),3,3), dtype=np.float )

for i in range(0,len(labels)):
    
    image_matrix = np.loadtxt(traindir+'/'+names[i])
    traindata[i] = image_matrix


######read test data from the test directory
testdir = sys.argv[2]

df = pd.read_csv(testdir+'/data.csv')
tnames = df['Name'].values


testdata = np.empty((len(labels),3,3), dtype=np.float)

for i in range(0,len(labels)):
    
    test_image_matrix = np.loadtxt(testdir+'/'+tnames[i])
    testdata[i] = test_image_matrix

sigmoid = lambda x: 1/(1+ np.exp(-x))
##############################
### Initialize all weights ###
c = np.ones((2,2), dtype=np.float)

epochs = 1000
eta = 0.1
prevobjective = np.inf
i=0

# calculate objective
objective = 0 
for i in range(0,len(labels)):
    hidden_layer = signal.convolve2d(traindata[i],c, mode='valid')
    for j in range(0,2,1):
        for k in range(0,2,1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    output_layer = (hidden_layer[0][0] + hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
    objective += (output_layer - labels[i])**2

# print("objective=", objective)

#gradient descent
stop = 0.01
while(prevobjective - objective >stop and i<epochs):

    #update previous objective
    prevobjective = objective
    
    dellc1 = 0
    dellc2 = 0
    dellc3 = 0
    dellc4 = 0
    f = (output_layer)**0.5

    for i in range(0, len(labels)):
                
        # convilution
        hidden_layer = signal.convolve2d(traindata[i],c,mode="valid")
        for j in range(0,2,1):
            for k in range(0,2,1):
                hidden_layer[j][k]= sigmoid(hidden_layer[j][k])

        
        ##calculate gradient for c1
        sqrtf = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1])/4 - labels[i]
        dz1dc1 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*traindata[i][0][0]
        dz2dc1 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*traindata[i][0][1]
        dz3dc1 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*traindata[i][1][0]
        dz4dc1 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*traindata[i][1][1]
        dellc1 += (sqrtf * (dz1dc1 + dz2dc1 + dz3dc1 +dz4dc1))/2
        ##calculate gradient for c2
       
        dz1dc2 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*traindata[i][0][1]
        dz2dc2 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*traindata[i][0][2]
        dz3dc2 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*traindata[i][1][1]
        dz4dc2 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*traindata[i][1][2]
        dellc2 += (sqrtf * (dz1dc2 + dz2dc2 + dz3dc2 +dz4dc2))/2
        ##calculate gradient for c3
        
        dz1dc3 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*traindata[i][1][0]
        dz2dc3 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*traindata[i][1][1]
        dz3dc3 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*traindata[i][2][0]
        dz4dc3 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*traindata[i][2][1]
        dellc3 += (sqrtf * (dz1dc3 + dz2dc3 + dz3dc3 +dz4dc3))/2
        ##calculate gradient for c4
       
        dz1dc4 = hidden_layer[0][0] *(1 - hidden_layer[0][0])*traindata[i][1][1]
        dz2dc4 = hidden_layer[0][1] *(1 - hidden_layer[0][1])*traindata[i][1][2]
        dz3dc4 = hidden_layer[1][0] *(1 - hidden_layer[1][0])*traindata[i][2][1]
        dz4dc4 = hidden_layer[1][1] *(1 - hidden_layer[1][1])*traindata[i][2][2]
        dellc4 += (sqrtf * (dz1dc4 + dz2dc4 + dz3dc4 +dz4dc4))/2

    #update c
    c[0][0] -= eta*dellc1
    c[0][1] -= eta*dellc2
    c[1][0] -= eta*dellc3
    c[1][1] -= eta*dellc4

    #recalculate objective
    objective = 0 
    for i in range(0,len(labels)):
       
        hidden_layer = signal.convolve2d(traindata[i],c, mode='valid')
       
        for j in range(0,2,1):
            for k in range(0,2,1):
                hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
        output_layer = (hidden_layer[0][0] + hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
        
        objective += (output_layer - labels[i])**2

    

print('\n'+"Convolutional kernel (c)="+'\n',c)

# predictions
print('\n'+"output_prediction=")
for i in range(0,len(labels)):
    hidden_layer = signal.convolve2d(testdata[i],c, mode='valid')
    for j in range(0,2,1):
        for k in range(0,2,1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    output_layer = (hidden_layer[0][0] + hidden_layer[0][1]+hidden_layer[1][0]+hidden_layer[1][1])/4
    if (output_layer < 0.5):
        print(-1)
    elif(output_layer >= 0.5):
        print(1)