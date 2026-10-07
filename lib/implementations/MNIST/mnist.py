"""Fully connected MNIST classifier from the original Python experiment."""

from ...experiment import run_training


def load_data():
    from tensorflow.keras.datasets import mnist

    return mnist.load_data()


def prepare_data(training, test):
    from tensorflow.keras.utils import to_categorical

    x_train, y_train = training
    x_test, y_test = test
    x_train = x_train.reshape(len(x_train), 784).astype("float32") / 255.0
    x_test = x_test.reshape(len(x_test), 784).astype("float32") / 255.0
    return (
        (x_train, to_categorical(y_train, 10)),
        (x_test, to_categorical(y_test, 10)),
    )


def build_model():
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(784,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def run(epochs=3):
    from tensorflow import keras

    keras.utils.set_random_seed(10)
    return run_training(load_data, prepare_data, build_model, epochs)
