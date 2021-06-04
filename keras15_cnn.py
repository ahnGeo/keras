from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2, 2), strides=1, input_shape=(5, 5, 1))) #Dense(10, input_shape=(5, 1))
model.add(Conv2D(5, (2, 2), padding='same'))

model.add(flatten())
model.add(Dense(1))

#model.summary()
