import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.
x = np.array([[10, 85, 70], [90, 85, 100], [80, 50, 30], [43, 60, 100]]) #(4, 3)
y = np.array([75, 65, 33, 85]) #(4, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1)

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape=(3,), activation='relu'))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))

#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4.
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# x_predict = np.transpose([[11, 12, 13], [21, 22, 23]])
# result = model.predict(x_predict)
# print("result : ", result)
