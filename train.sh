#!/usr/bin/env bash

exists() {
    # check if passed command exists on system
    command -v "${1}" >/dev/null 2>&1
}

train_with_pip() {
    .venv/bin/activate
    python src/main.py train best_model
}

train_with_poetry() {
    poetry run python src/main.py train best_model
}

main() {
    if exists poetry; then
        echo "Train with poetry"
        train_with_poetry
    else
        echo "Train with pip venv"
        train_with_pip
    fi
}

main