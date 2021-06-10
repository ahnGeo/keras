import numpy as np
from tensorflow.keras.datasets import cifar10

#1.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) #(50000, 32, 32, 3), (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, ) (10000, )

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)

#2.
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=50, kernel_size=(2, 2), padding='same', strides=1, input_shape=(32, 32, 3)))
model.add(Conv2D(20, (2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#3.
#model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics='acc')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)

#4.
results = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", results[0])
print("acc : ", results[1])

