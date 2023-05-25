#!/usr/bin/env bash
#
# Setup python environment by using poetry or venv and pip to install dependencies
# This should be run from the ROOT DIRECTORY of the project

exists() {
    # check if passed command exists on system
    command -v "${1}" >/dev/null 2>&1
}

install_with_pip() {
    # install dependencies using venv and pip
    # make sure to use python ^3 by checking the default interpreter
    major_version="$(python -c 'import sys; print(sys.version_info[0])')"
    if [[ "$major_version" == 3 ]]; then
        python -m venv .venv
    else
        echo "Default version of python is python${major_version}, using python3 instead"
        python3 -m venv .venv
    fi

    .venv/bin/activate
    pip install -r requirements.txt
}

install_with_poetry() {
    # install dependencies using poetry
    poetry install
}

install_dependencies() {
    # check if poetry is installed
    if exists poetry; then
        echo "Using poetry to install dependencies"
        install_with_poetry
    else
        echo "Could not find poetry, using pip to install dependencies"
        install_with_pip
    fi
}

main () {
    echo "Setup python environment"
    install_dependencies
}

main