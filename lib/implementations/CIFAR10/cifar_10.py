# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import time

np.random.seed(10)
tf.compat.v1.disable_eager_execution()

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
np.random.seed(10)


def plot_sample(x, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])


def plot_range(x_train, y_train, classes):
    num_classes = 10
    f, ax = plt.subplots(1, num_classes, figsize=(32, 32))

    for i in range(0, num_classes):
        sample = x_train[i]
        ax[i].imshow(sample)
        ax[i].set_title("Label: {}".format(classes[y_train[i]]), fontsize=16)


"""# DATA """
print("\n\nGetting data")
# Classes:
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck
# image size: 32x32, RGB (colors = 3)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = y_test.reshape(-1, )  # Transposta
y_train = y_train.reshape(-1, )

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

"""# Analysing data"""
print("\n\nAnalysing data")
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

plot_sample(x_train, y_train, 7)
plot_range(x_train, y_train, classes)

"""# Prepare Data"""

print("\n\nPrepare Data")
# Normalize Data
x_train = x_train / 255.0
x_test = x_test / 255.0

"""# Model - FCNN """

print("\n\nModel - FCNN ")
# Modelo BOM CNN
model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
# model.build(input_shape)
model.summary()

"""# Train"""
print("\n\nTrain")
start_training_time = time.time()
epochs = 1
model.fit(x=x_train, y=y_train, epochs=epochs)
end_training_time = time.time()
print("Total training time using {0} epochs: {1} seconds".format(epochs, end_training_time - start_training_time))

"""# Evaluate"""
print("\n\nEvaluate")
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
# Creating classes array
y_pred_classes_str = []
for i in y_pred_classes:
    y_pred_classes_str.append(classes[i])

print(y_pred, y_pred_classes)

random_idx = np.random.choice(len(x_test))
x_sample = x_test[random_idx]

y_sample_true = classes[y_test[random_idx]]
y_sample_pred_class = y_pred_classes_str[random_idx]

plt.title("Predicated: {}, True: {}".format(y_sample_pred_class, y_sample_true, frozenset=16))
plt.imshow(x_sample.reshape(32, 32, 3))
