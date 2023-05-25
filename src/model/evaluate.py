import csv
import datetime
from pathlib import Path

from tensorflow import keras
from bert.tokenization.bert_tokenization import FullTokenizer

import sklearn
import numpy as np

from data_preparation.facebook_comments import FacebookComments
from model import create_model

OUTPUT_DIRECTORY = "outputs"
CHECKPOINT_DIRECTORY = "model/multi_cased_L-12_H-768_A-12"
CHECKPOINT_FILE = "bert_model.ckpt"
CONFIG_FILE = "bert_config.json"
VOCAB_FILE = "vocab.txt"


def load_model(
        model_name,
        lstm_units=768,
        dense_units=768,
        dropout=0.5,
        adapter_size=64,
        optimizer=None,
        bert_model_path=None
):
    """Creates model and loads weights.

        Returns:
            trained model and processed data"""

    optimizer = optimizer or keras.optimizers.Adam()
    bert_model_path = bert_model_path or CHECKPOINT_DIRECTORY

    tokenizer = FullTokenizer(vocab_file=str(Path(bert_model_path, VOCAB_FILE)))
    data = FacebookComments(
        tokenizer,
        sample_size=10 * 128 * 2,
        max_seq_len=128,

    )
    model = create_model(
        str(Path(bert_model_path, CONFIG_FILE)),
        str(Path(bert_model_path, CHECKPOINT_FILE)),
        max_seq_len=data.max_seq_len,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout,
        adapter_size=adapter_size,
        optimizer=optimizer
    )

    model.load_weights(
        f"./{OUTPUT_DIRECTORY}/{model_name}.h5")

    return model, data


def predict(model_name,
            lstm_units=768,
            dense_units=768,
            dropout=0.5,
            adapter_size=64,
            optimizer=None,
            bert_model_path=None):
    optimizer = optimizer or keras.optimizers.Adam()
    bert_model_path = bert_model_path or CHECKPOINT_DIRECTORY

    model, data = load_model(
        model_name,
        lstm_units,
        dense_units,
        dropout,
        adapter_size,
        optimizer,
        bert_model_path)

    return model.predict(data.dev_x), model.predict(data.test_x)


def evaluate(model, x, y):
    """Evaluates the model and saves results.

    Args:
        model: trained model
            x: list of processed sentences
            y: list of gold labels

    Returns:
        accuracy and f1-score
    """

    y_pred1 = model.predict(x)
    y_pred = np.argmax(y_pred1, axis=1)

    _, accuracy = model.evaluate(x, y)

    f1_score = sklearn.metrics.f1_score(y_pred, y, average='macro')
    return accuracy, f1_score


if __name__ == '__main__':
    model, data = load_model()
    dev_accuracy, dev_f1_score = evaluate(model, data.dev_x, data.dev_y)
    test_accuracy, test_f1_score = evaluate(model, data.test_x, data.test_y)

    print(f"dev-accuracy: {dev_accuracy}, dev-f1-score: {dev_f1_score}\
    \n test-accuracy: {test_accuracy}, test-f1-score: {test_f1_score}")
    with open(f'{OUTPUT_DIRECTORY}/results.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["dev-accuracy", dev_accuracy])
        tsv_writer.writerow(["dev-f1-score", dev_f1_score])
        tsv_writer.writerow(["test-accuracy", test_accuracy])
        tsv_writer.writerow(["test-f1-score", test_f1_score])
