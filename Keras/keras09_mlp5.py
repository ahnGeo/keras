#다 : 다 mlp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(711, 811), range(1, 101), range(201, 301)])

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.8, random_state=66) #랜덤 계수(random_state) = shuffle 순서 고정

#2.
model = Sequential()
model.add(Dense(10, input_shape=(3,), activation='relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))


#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=60, batch_size=1)

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