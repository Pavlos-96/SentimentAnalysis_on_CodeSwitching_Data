#!/usr/bin/env bash
#
# Setup MODEL_NAME model downloaded from MODEL_URL in MODEL_PATH
# This should be run from the ROOT DIRECTORY of the project

MONO_MODEL_NAME="uncased_L-12_H-768_A-12"
MONO_MODEL_URL="https://storage.googleapis.com/bert_models/2020_02_20/${MONO_MODEL_NAME}.zip"

MULTI_MODEL_NAME="multi_cased_L-12_H-768_A-12"
MULTI_MODEL_URL="https://storage.googleapis.com/bert_models/2018_11_23/${MULTI_MODEL_NAME}.zip"

MODEL_PATH="model"

provide_model() {
    # Download and unzip model.
    # Place model in MODEL_PATH and delete .zip file.
    model_url=$1
    model_name=$2

    wget ${model_url}
    unzip ${model_name}.zip
    mv ${model_name} ${MODEL_PATH}/${model_name}
    rm ${model_name}.zip
}

prepare_dir() {
    # Make sure MODEL_PATH exists
    # Create if not existant
    if [[ ! -d "${MODEL_PATH}" ]]; then
        mkdir ${MODEL_PATH}
    fi
}

main() {
    echo "Setup model"
    prepare_dir
    # provide_model ${MONO_MODEL_URL} ${MONO_MODEL_NAME}
    provide_model ${MULTI_MODEL_URL} ${MULTI_MODEL_NAME}
}

main
