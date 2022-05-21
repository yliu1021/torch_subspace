# Torch Subspace

## Code Formatting
Run the `format.sh` script to format all files
```
./format.sh
```

## Packages

### `torch_subspace`
This contains the code for low rank subspace layers

### `tests`
Unit tests for `torch_subspace`

### `pruners`
Pruning algorithms

### `blockers`
Blocking algorithms

### `experiments`
Helper functions and modules for experiments

### `main.py`
Run this file for experiments

## Unit Tests
To run unittests:
```
python -m unittest tests
```

To create a unittest, create a module in the `tests` package and export it in the `__init__.py`

**Make sure to export it in `tests/__init__.py`. Otherwise, it won't get picked up**

### Coverage
First install coverage
```
pip install coverage
```

Then run
```
coverage run -m unittest tests
```

To see coverage results, run
```
coverage report
```

Optionally, HTML coverage can be generated using
```
coverage html
```
and then viewed at `htmlcov/index.html`
