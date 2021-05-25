import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import array

#1.
x = np.array(range(1, 101))
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True) #default : shuffle=True
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.1)


#2.
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=60, 
    validation_data=(x_val, y_val)
    , batch_size=1)

#4.
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

result = model.predict([101, 102, 103])
print("result : ", result)
