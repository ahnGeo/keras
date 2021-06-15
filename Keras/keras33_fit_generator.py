import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#
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

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
model = Sequential()
model.add(Conv2D(32, (3, 3)))
model.add(Flatten())
model.add(1, activation='sigmoid')

#3.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(xy_train, steps_per_epoch=206, epochs=10, #stepsperepoch = 전체 / 배치
    validation_data=xy_test) 
results = model.evaluate_generator(xy_train)
print(results)