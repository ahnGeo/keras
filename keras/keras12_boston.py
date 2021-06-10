import numpy as np

#sklearn boston
from sklearn.datasets import load_boston

#1.
dataset = load_boston()
x = dataset.data
y = dataset.target
print("shape of boston dataset : ", x.shape, y.shape)
#(506, 13) (506,)

# print(dataset.feature_names)
# print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.8, random_state=66)

#2.
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
x1 = Dense(100)(input1)
x2 = Dense(200)(x1)
x3 = Dense(50)(x2)
x5 = Dense(10)(x3)
output1 = Dense(1)(x5)
model = Model(inputs=input1, outputs=output1)


#3.
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=300, batch_size=1,
    verbose=2, validation_split=0.1)

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