import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import  utils
import numpy as np
import pandas as pd

df = pd.read_csv('Iris.csv')

x = df.values[0:, 1:5]

y = df.values[0:, 5:]

j = 0
for i in y:
    if i == "Iris-setosa":
        y[j, [0]] = 0
    elif i == "Iris-versicolor":
        y[j, [0]] = 1
    else:
        y[j, [0]] = 2
    j += 1

model = Sequential()

model.add(Dense(32, activation='tanh', input_shape=(4,)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

y = utils.to_categorical(y, 3)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x,y,epochs=2000)

model.save('model.h5')