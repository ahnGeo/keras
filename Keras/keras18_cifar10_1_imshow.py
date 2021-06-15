import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) #(50000, 32, 32, 3), (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, ) (10000, )

#시각화
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()

