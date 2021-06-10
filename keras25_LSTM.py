#RNN

import numpy as np

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]) #(4, 5)
y = np.array([4, 5, 6, 7]) #(4,)

x = x.reshape(4, 3, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=(3, 1))) #LSTM에 input은 3차원, input_shape=2D, Dense는 2차원, 이미지는 4차원
                                   # 몇 개씩 자르는지
model.add(Dense(10))
model.add(Dense(1))

model.summary() # -> parameter 이유 과제