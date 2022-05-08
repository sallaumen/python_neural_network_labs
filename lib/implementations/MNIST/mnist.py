# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
from keras.datasets import mnist

np.random.seed(10)

"""# Data"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

"""# Visualize Examples"""

num_classes = 10

y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

"""# Prepare Data"""

# Normalize Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape Data
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

"""# Create Model - Fully Connected Neural Network"""

model = Sequential()

model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

"""# Train"""
print("\n\nTrain")

start_training_time = time.time()
epochs = 3
model.fit(x=x_train, y=y_train, epochs=epochs)
end_training_time = time.time()
print("Total training time using {0} epochs: {1} seconds".format(epochs, end_training_time - start_training_time))

"""# Evaluate"""
print("Evaluating")

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

# y_pred = model.predict(x_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# print(y_pred)
# print(y_pred_classes)
