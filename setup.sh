#!/usr/bin/env bash
#
# Setup all requirements needed to run train.sh, predict.sh and evaluate.sh
# This should be run from the ROOT DIRECTORY of the project

setup() {
    ./src/bin/setup_environment.sh
    ./src/bin/setup_bert_model.sh
}

setup