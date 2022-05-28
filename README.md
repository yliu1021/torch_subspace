# Torch Subspace

## Experiments
Running guild experiments:

First start background queues specifying visible GPUs:
```
for i in {0..7}; do guild run queue -b --gpus="$i" -y; done
```
Then stage trials
```
guild run main.py \
model="vgg16" \
save_path="/home/yliu/torch_subspace/checkpoint/vgg16_cifar10_epoch_160" \
dataset="cifar10" \
data_location="~/torch_subspace/data/" \
blocker='["square","alds","none"]' \
pruner='["alignment_output","alignment_output_sampling","alignment_output_sampling_proportional","relative_error","magnitude"]' \
sparsity='[0.99,0.98,0.96,0.93,0.89]' \
preprune_epochs=160 \
postprune_epochs=160 \
--stage-trials \
--label="vgg16_cifar10"
```

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
