import os
import numpy as np
import pandas as pd
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras.utils import np_utils
from keras import applications
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

img_width = 150
img_height = 150

data_dir = sys.argv[1]
model_file_name = sys.argv[2]

train_datagen = ImageDataGenerator(
        validation_split=0.4)

validation_generator = train_datagen.flow_from_directory(
        data_dir, #same as in train generator
        target_size=(img_width, img_height),
        subset='validation')

model = load_model(model_file_name)
model.summary()
# evaluate the model
scores = model.evaluate_generator(validation_generator, steps=50)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
