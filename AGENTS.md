# Contributor and AI agent guidance

This repository holds the Python half of a historical capstone comparison. Read `README.md` and the [comparison overview](https://github.com/sallaumen/elixir_vs_python_nn_performance_comparison) before changing experiment behavior.

- Keep `lib/MVP/` as archived source material. Make maintained code changes in `lib/implementations/`, `lib/experiment.py`, and `lib/main.py`.
- Keep imports free of dataset downloads and training. The CLI should perform work only after a dataset is selected.
- Preserve each model's layer order, optimizer, loss, default epochs, preprocessing, and timing boundary unless the change explicitly states why the experiment changes.
- Evaluate on the dataset's test split. Never report training accuracy as test accuracy or compare times without recording hardware, software versions, dataset, batch size, epochs, and timing scope.
- Add focused tests for data or orchestration changes. Run `python -m unittest discover -s tests` and `python -m compileall -q lib tests`. Full training requires TensorFlow and dataset downloads.
- Write comments, docs, commit messages, and user-facing output in clear English. Do not claim parity with the Elixir models; document differences.
