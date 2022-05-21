#!/bin/bash
set -e
SRC=( \
    blockers \
    experiments \
    pruners \
    tests \
    torch_subspace \
    main.py \
)

isort $SRC
black $SRC
