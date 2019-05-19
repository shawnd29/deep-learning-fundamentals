from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import sys
import numpy as np
import pandas as np
from keras.models import load_model

# dimensions of our images.
img_width, img_height = 224, 224
batch_size = 16


test_data_dir = sys.argv[1]
model_file_name = sys.argv[2]


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model = load_model(model_file_name)
model.summary()
# evaluate the model
scores = model.evaluate_generator(test_generator, steps=50)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
