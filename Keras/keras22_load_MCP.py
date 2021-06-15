import re
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. Data
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66)

#2. Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

model = load_model('C:\Keras\keras\checkpoint\k21_cancer_100-0.5045.hdf5')

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])