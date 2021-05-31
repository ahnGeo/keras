#함수형 모델
#다 : 1
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(301, 401)])
y = np.array(range(711, 811))

x = np.transpose(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.8, random_state=66) #랜덤 계수(random_state) = shuffle 순서 고정

#2.
# model = Sequential()
# model.add(Dense(3, input_shape=(5,), activation='relu'))
# model.add(Dense(4))
# model.add(Dense(2))
# model.summary()

input1 = Input(shape=(5,))
x1 = Dense(3)(input1)
x1 = Dense(7)(x1)
x1 = Dense(4)(x1)
output1 = Dense(1)(x1)

model = Model(inputs=input1, outputs=output1)
model.summary()

#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=60, batch_size=1,
    verbose=0)

#verbose=0 : 안 보임
#verbose=1 : 보임
#verbose=2 : progress bar 안 보임
#verbose=3, 4, 5, ... : epoch 수만 나옴

#4.
loss = model.evaluate(x_test, y_test)
print("results : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))    
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_predict, y_test)
print("R2 : ", R2)