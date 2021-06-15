import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#이미지에 변화 주기 -> 증폭
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    './tmp/horse-or-human',
    target_size=(300, 300),
    batch_size=5,
    class_mode='binary'
)

xy_test = test_datagen.flow_from_directory(
    './tmp/testdata',
    target_size=(300, 300),
    batch_size=5,
    class_mode='binary'
)