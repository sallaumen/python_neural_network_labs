# Python neural network experiments

Python implementations of the MNIST and CIFAR-10 image classifiers from Lucas Campos Tavano's Computer Engineering capstone project at UTFPR. This repository contains runnable training scripts and preserves the original notebooks and figures under [`lib/MVP/`](lib/MVP/).

The [comparison overview](https://github.com/sallaumen/elixir_vs_python_nn_performance_comparison) links this repository to the Elixir implementation and explains the limits of comparing their historical results.

## Experiments

| Dataset | Model | Optimizer | Epochs | Test evaluation |
| --- | --- | --- | ---: | --- |
| MNIST | Dense 128 → Dense 128 → Dropout 0.5 → Dense 10 | Adam | 3 by default | Categorical cross-entropy and accuracy |
| CIFAR-10 | Conv 32 → Pool → Conv 64 → Pool → Dense 64 → Dense 10 | SGD | 3 by default | Sparse categorical cross-entropy and accuracy |

Each CLI run sets the Keras random seed to 10. Input pixels are scaled to `[0, 1]`. MNIST images are flattened to 784 features; CIFAR-10 images retain their 32 × 32 × 3 shape. The timer measures training (`model.fit`) only. Evaluation uses the datasets' separate test splits after training.

## Setup

Use a Python version supported by your TensorFlow release. See the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip) for platform and GPU requirements.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the dataset. CPU execution is supported; GPU setup depends on your platform and TensorFlow installation.

## Run

From the repository root:

```sh
python -m lib.main mnist
python -m lib.main cifar10 --epochs 3
```

The commands print training duration, test loss, and test accuracy. Use `python -m lib.main --help` for CLI options. The archived notebooks may require extra plotting and analysis libraries; they are historical artifacts, not the maintained CLI.

## Checks

```sh
python -m unittest discover -s tests
python -m compileall -q lib tests
```

These quick checks do not download datasets or run TensorFlow training. For a full validation, run each command above in an environment with TensorFlow installed and record the Python, TensorFlow, hardware, and dataset versions alongside results.

## Historical figures

The figures below come from the original project. They are not measurements produced by the current CLI.

| Dataset | Recorded performance | Example prediction |
| --- | --- | --- |
| MNIST | [Performance](lib/MVP/MNIST/test_performance.png) | [Prediction](lib/MVP/MNIST/prediction.png) |
| CIFAR-10 | [Performance](lib/MVP/CIFAR10/test_performance.png) | [Prediction](lib/MVP/CIFAR10/prediction.png) |

See the [MNIST notebook](lib/MVP/MNIST/MNIST_MVP.ipynb) and [CIFAR-10 notebook](lib/MVP/CIFAR10/CIFAR10_MVP.ipynb) for the original exploratory work.
