import numpy as np
import tensorflow as tf


#data
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#evaluate, predict
loss = model.evaluate(x, y, batch_size=1)
print('loss : ', loss)

results = model.predict([4])
print('results : ', results)

#훈련 양 늘리기 : 배치사이즈 줄이기, epoch 늘리기, 노드 수 늘리기


