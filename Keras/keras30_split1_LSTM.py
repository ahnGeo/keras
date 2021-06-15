import numpy as np

#1.
a = np.array(range(1, 11))
size = 6
print(a)

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
x = dataset[:, :(size-1)]
y = dataset[:, size-1]
print(x)

x = x.reshape(5, 5, 1)

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100, input_shape=(5, 1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

#4.
results = model.evaluate(x, y, batch_size=1)

print("loss : ", results)

x_pred = [[6, 7, 8, 9, 10]]
x_pred = np.array(x_pred)
x_pred = x_pred.reshape(1, 5, 1)
y_pred = model.predict(x_pred)
print(y_pred)
#10.87

