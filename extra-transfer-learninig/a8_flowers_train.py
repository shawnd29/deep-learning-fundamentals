import os
import numpy as np
import pandas as pd
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras.utils import np_utils
from keras import applications
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

img_width = 150
img_height = 150
num_classes = 5
data_dir = sys.argv[1]
model_file_name = sys.argv[2]

train_datagen = ImageDataGenerator(
        validation_split=0.4)

train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        data_dir, #same as in train generator
        target_size=(img_width, img_height),
        subset='validation')

# Define training params
batch_size = 400
epochs = 1 # Set to 1 for demo, for good results set 100, Running for 100 epochs will take time.
train_steps_per_epoch = train_generator.n//train_generator.batch_size
val_steps_per_epoch = validation_generator.n//validation_generator.batch_size

#weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg16_model = applications.VGG16(include_top=False, weights='imagenet')

model_top = Sequential()
model_top.add(Flatten(input_shape=(img_width, img_height, 3)))
model_top.add(Dense(512, activation='relu'))
model_top.add(Dense(256, activation='relu'))
model_top.add(Dense(128, activation='relu'))
model_top.add(Dense(64, activation='relu'))
model_top.add(Dense(32, activation='sigmoid'))
model_top.add(Dropout(0.50))
model_top.add(Dense(num_classes, activation='softmax'))

model_top.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

print (model_top.summary())

model_top.fit_generator(generator=train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=val_steps_per_epoch,
                    epochs=epochs)


model_top.save(model_file_name + '.h5')
