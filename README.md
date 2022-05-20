# Torch Subspace

## Code Formatting
`isort` and `black` are used for code formatting. Those can be installed via
```
pip install isort black
```

To run them:
```
isort experiments pruners tests torch_subspace main.py && black experiments pruners tests torch_subspace main.py
```

This can be added to a git hook:
```
echo "isort experiments pruners tests torch_subspace main.py && black experiments pruners tests torch_subspace main.py" > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

## Packages

### `torch_subspace`
This contains the code for low rank subspace layers

### `tests`
Unit tests for `torch_subspace`

### `pruners`
Pruning algorithms

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
