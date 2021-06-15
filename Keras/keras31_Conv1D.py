#Conv1D
#input shape = (28, 28)로 풀어라
#Conv2D (X)
#LSTM -> so slow
#Conv1D

import numpy as np
from tensorflow.keras.datasets import mnist

#1.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) #(60000, 28, 28), (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000, ) (10000, )

#no reshape only regularization
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2.
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2, padding='same', strides=1, input_shape=(28, 28)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(10, activation='softmax'))

#3.
#model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics='acc')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)

#4.
results = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", results[0])
print("acc : ", results[1])
