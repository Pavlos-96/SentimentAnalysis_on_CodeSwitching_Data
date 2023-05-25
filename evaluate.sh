#!/usr/bin/env bash

exists() {
    # check if passed command exists on system
    command -v "${1}" >/dev/null 2>&1
}

evaluate_with_pip() {
    .venv/bin/activate
    python src/main.py evaluate best_model
}

evaluate_with_poetry() {
    poetry run python src/main.py evaluate best_model
}

main() {
    if exists poetry; then
        echo "Evaluate with poetry"
        evaluate_with_poetry
    else
        echo "Evaluate with pip venv"
        evaluate_with_pip
    fi
}

main