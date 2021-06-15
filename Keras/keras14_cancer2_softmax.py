#이진 분류
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

#1.
datasets = load_breast_cancer()

x = datasets.data #(569, 30)
y = datasets.target #(569,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
#print(y.shape) #(569, 2)

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(2, activation='softmax'))

#3
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x, y, epochs=70, batch_size=1, validation_split=0.1)

results = model.evaluate(x, y)
print('loss : ', results[0])
print('metrics : ', results[1])

y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])
