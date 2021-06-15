import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import array

#1.
x = np.array(range(1, 101))
y = np.array(range(101, 201))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

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
