import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.
x = np.arange(1, 11)
y = np.array([1, 2, 4, 3, 5, 5, 7, 9, 8, 11])

#2.
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4.
y_pred = model.predict(x)
print(y)
print(y_pred)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

#         x   y
# train   o   o
# validation o   o
# test    o   o
# predict o   x