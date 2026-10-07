"""Command-line entry point for the Python experiments."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a historical image classifier")
    parser.add_argument("dataset", choices=("mnist", "cifar10"))
    parser.add_argument("--epochs", type=int, default=3, help="training epochs (default: 3)")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")

    if args.dataset == "mnist":
        from .implementations.MNIST.mnist import run
    else:
        from .implementations.CIFAR10.cifar_10 import run

    result = run(epochs=args.epochs)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Training time: {result['training_seconds']:.3f} s ({args.epochs} epochs)")
    print(f"Test loss: {result['test_loss']:.4f}")
    print(f"Test accuracy: {result['test_accuracy']:.2%}")
    return result


if __name__ == "__main__":
    main()
