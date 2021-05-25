#multi layer perceptron
#perceptron : input -> weight * x + b -> activation function -> y hat ><

import numpy as np
#1.
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15,16,17,18,19,20]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = np.transpose(x) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1)

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape=(2,), activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=60, batch_size=1)

#4.
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

x_predict = np.transpose([[11, 12, 13], [21, 22, 23]])
result = model.predict(x_predict)
print("result : ", result)
