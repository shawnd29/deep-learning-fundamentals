from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import sys
import numpy as np
import pandas as np


# dimensions of our images.
img_width, img_height = 150, 150


train_data_dir = sys.argv[1]
model_file_name = sys.argv[2]
validation_data_dir = '/home/u/usman/cs_677_datasets/chest_xray/val'
test_data_dir = '/home/u/usman/cs_677_datasets/chest_xray/test'

nb_train_samples = 5217
nb_validation_samples = 17
epochs = 1
batch_size = 16


if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.layers
model.input
model.output

model.compile(loss='binary_crossentropy',
			  optimizer='rmsprop',
			  metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

test_generator = test_datagen.flow_from_directory(
	test_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples, #// batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples )#// batch_size)

model.save(model_file_name + '.h5')

# evaluate the model
scores = model.evaluate_generator(test_generator, steps=50)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


