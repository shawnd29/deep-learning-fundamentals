#This program is to train the Mini-ImageNet convolution 

#To run this program 
#KERAS_BACKEND=tensorflow python a6_train.py  x_train y_train model_1 

#Done by Shawn D'Souza - srd59

 

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras import backend as K
K.set_image_dim_ordering('th') #Used to fix the dimension error
import numpy as np
import keras
import sys
batch_size = 64
nb_classes = 10
nb_epoch = 100

img_channels = 3
img_rows = 32
img_cols = 32

#Read Data

X_train = np.load(sys.argv[1])
X_test = np.load('x_test.npy')
Y_train = np.load(sys.argv[2])
Y_test = np.load('y_test.npy')

#Read Data
#X_train = np.load(sys.argv[1])
#X_test = np.load(sys.argv[2])
#Y_train = np.load(sys.argv[3])
#Y_test = np.load(sys.argv[4])


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

Y_train = to_categorical(Y_train, nb_classes)
Y_test = to_categorical(Y_test, nb_classes)#convert label into one-hot vector

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

model = Sequential()

#Max pooling was used throughout as it was the quickest to compute

#Layer 1
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', input_shape=[3,112,112]))#Convo$
model.add(Activation('relu'))#Activation function
model.add(MaxPooling2D(pool_size=(2, 2)))
#14x14 output

#Layer 2
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid'))#Convo$
model.add(Activation('relu'))#Activation function
model.add(MaxPooling2D(pool_size=(2, 2)))
#6x6

#Layer 3
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid'))#Convo$
model.add(Activation('relu'))#Activation function
model.add(MaxPooling2D(pool_size=(2, 2)))
#2x2

#Dense layer
model.add(Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model.add(Dropout(0.5))
model.add(Dense(10))#Fully connected layer
model.add(Activation('softmax'))

#Adam optimizer was used as to 
opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def train():
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=(X_test,Y_test),
              shuffle=True)

train()
model.save(sys.argv[3]+'.h5')
