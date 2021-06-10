from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = load_model('./keras/model/k23_1_model_1.h5')

model.summary()