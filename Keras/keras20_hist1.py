#이진 분류
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

#1.
datasets = load_breast_cancer()

x = datasets.data #(569, 30)
y = datasets.target #(569,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(2, activation='softmax'))

#3
#sparser_categorical_crossentropy 사용하면 원핫인코딩 안 해도 해 줌
#그래서 output 2로 맞춰줘야 함
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
hist = model.fit(x_train, y_train, epochs=300, batch_size=1, validation_split=0.1, 
            callbacks=early_stopping)

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('metrics : ', results[1])

y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])

print(hist)
print(hist.history.keys())
#print(hist.history['loss'])
print(hist.history['loss'])
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'])
plt.show()