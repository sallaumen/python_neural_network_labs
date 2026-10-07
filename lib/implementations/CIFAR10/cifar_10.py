"""Convolutional CIFAR-10 classifier from the original Python experiment."""

from ...experiment import run_training


def load_data():
    from tensorflow.keras.datasets import cifar10

    return cifar10.load_data()


def prepare_data(training, test):
    x_train, y_train = training
    x_test, y_test = test
    return (
        (x_train.astype("float32") / 255.0, y_train.reshape(-1)),
        (x_test.astype("float32") / 255.0, y_test.reshape(-1)),
    )


def build_model():
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(32, 32, 3)),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="SGD", metrics=["accuracy"]
    )
    return model


def run(epochs=3):
    from tensorflow import keras

    keras.utils.set_random_seed(10)
    return run_training(load_data, prepare_data, build_model, epochs)
