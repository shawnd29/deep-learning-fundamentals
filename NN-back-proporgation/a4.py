 #This program finds the predictions using a single layeer nwural network of 3 nodes while using minibatch
# To run this program type
# python a3.py <train> <test>

# Done by shawn D'Souza

import numpy as np
import sys

#################
### Read data ###

f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

rows = train.shape[0]
cols = train.shape[1]

onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)
hidden_nodes = 3

##############################
### Initialize all weights ###

w = np.random.rand(hidden_nodes)
W = np.random.rand(hidden_nodes, cols)

epochs = 100
eta = .001
prevobj = np.inf
i=0

###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])

output_layer = np.matmul(hidden_layer, np.transpose(w))

obj = np.sum(np.square(output_layer - trainlabels))

###############################
### Begin gradient descent ####
mb_size=10
while(i < epochs):
		
	#Update previous objective
	prevobj = obj
	rowindices=np.array([i for i in range(rows)])
	np.random.shuffle(rowindices)
	dellw = (np.dot(hidden_layer[0,:],w)-trainlabels[0])*hidden_layer[0,:]
	for j in range(0,mb_size,1):
		index=rowindices[j]
		dellw += (np.dot(hidden_layer[index,:],np.transpose(w))-trainlabels[index])*hidden_layer[index,:]

		#Update w
	w = w - eta*dellw
	#dells
	dells = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[0] * (hidden_layer[0,0])*(1-hidden_layer[0,0])*train[0]
	for j in range(1, rows):
		dells += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[0] * (hidden_layer[j,0])*(1-hidden_layer[j,0])*train[j]
	
	for k in range(0,rows,1):
		index=rowindices[k]
		dells = np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[0] * (hidden_layer[index,0])*(1-hidden_layer[index,0])*train[index]
		for j in range(0,mb_size,1):
			index=rowindices[j]
			dells += np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[0] * (hidden_layer[index,0])*(1-hidden_layer[index,0])*train[index]
	#dellu
	for k in range(0,rows,1):
		index=rowindices[k]
		dellu = np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[1] * (hidden_layer[index,1])*(1-hidden_layer[index,1])*train[index]
		for j in range(0,mb_size,1):
			index=rowindices[j]
			dellu += np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[1] * (hidden_layer[index,1])*(1-hidden_layer[index,1])*train[index]
		#dellv
	for k in range(0,rows,1):
		index=rowindices[k]
		dellv = np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[2] * (hidden_layer[index,2])*(1-hidden_layer[index,2])*train[index]
		for j in range(0,mb_size,1):
			index=rowindices[j]	
			dellv += np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[2] * (hidden_layer[index,2])*(1-hidden_layer[index,2])*train[index]
		
	dellW=np.array ([dells,dellu,dellv])
	#Update W
	W = W - eta*dellW

	#Recalculate objective
	hidden_layer = np.matmul(train, np.transpose(W))
	
	hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
	
	output_layer = np.matmul(hidden_layer, np.transpose(w))
	
	obj = np.sum(np.square(output_layer - trainlabels))

		
	i = i + 1
	print("Objective=",obj)


### Do final predictions ###

x=np.matmul(test,np.transpose(W))
predictions =np.sign(np.matmul(sigmoid(x), np.transpose(w)))
print("final prediction",predictions)
#print(w)