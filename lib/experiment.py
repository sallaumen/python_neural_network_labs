"""Shared training workflow for the two historical image classifiers."""

from time import perf_counter


def run_training(load_data, prepare_data, build_model, epochs):
    """Train once and evaluate on the dataset's separate test split.

    Only ``model.fit`` is timed. Dataset download, preprocessing, model
    construction, and evaluation are intentionally outside the timer.
    """
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")

    training, test = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data(training, test)
    model = build_model()

    start = perf_counter()
    model.fit(x_train, y_train, epochs=epochs, verbose=2)
    training_seconds = perf_counter() - start

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return {
        "training_seconds": training_seconds,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "training_samples": len(x_train),
        "test_samples": len(x_test),
    }
