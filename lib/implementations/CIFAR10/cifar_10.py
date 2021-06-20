# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import cifar10

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

"""#DATA"""
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
model = models.Sequential()
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics='accuracy')
model.summary()

"""# Train"""
print("\n\nTrain")
model.fit(x=x_train, y=y_train, epochs=10)

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

"""# Investigate Some Errors"""

def investigate_errors():
    global i
    print(y_pred_classes, y_true)
    errors = (y_pred_classes != y_true)
    y_pred_classes_errors = y_pred_classes[errors]
    y_pred_errors = y_pred[errors]
    y_true_errors = y_true[errors]
    x_test_errors = x_test[errors]
    y_pred_errors_probability = np.max(y_pred_errors, axis=1)
    true_probability_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
    diff_errors_pred_true = y_pred_errors_probability - true_probability_errors
    # Get list of indices of sorted differences
    sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)
    top_idx_diff_errors = sorted_idx_diff_errors[-5:]  # 5 last ones
    # Show Top Errors
    num = len(top_idx_diff_errors)
    f, ax = plt.subplots(1, num, figsize=(30, 30))
    for i in range(0, num):
        idx = top_idx_diff_errors[i]
        sample = x_test_errors[idx].reshape(28, 28)
        y_t = y_true_errors[idx]
        y_p = y_pred_classes_errors[idx]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title("Predicted label :{}\nTrue label: {}".format(y_p, y_t), fontsize=22)

# investigate_errors()
