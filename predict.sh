#!/usr/bin/env bash

exists() {
    # check if passed command exists on system
    command -v "${1}" >/dev/null 2>&1
}

predict_with_pip() {
    .venv/bin/activate
    python src/main.py predict best_model
}

predict_with_poetry() {
    poetry run python src/main.py predict best_model
}

main() {
    if exists poetry; then
        echo "Predict with poetry"
        predict_with_poetry
    else
        echo "Predict with pip venv"
        predict_with_pip
    fi
}

main