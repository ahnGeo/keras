import numpy as np
#1.
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15])
y_test = np.array([11, 12, 13, 14, 15])

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape=(1,), activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=80, validation_split=0.1, batch_size=1)

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
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)