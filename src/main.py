import argparse
import csv

from model import train, evaluate, load_model, predict

if __name__ == '__main__':
    # parameters
    BATCH_SIZE = 48
    EPOCH_COUNT = 50
    WARMUP_EPOCH_COUNT = 20
    MAX_LEARN_RATE = 1e-5
    END_LEARN_RATE = 1e-7
    PATIENCE = 20
    LSTM_UNITS = 768
    DENSE_UNITS = 768
    DROPOUT = 0.5
    ADAPTER_SIZE = None
    OPTIMIZER = None
    BERT_MODEL_PATH = "model/multi_cased_L-12_H-768_A-12"

    OUTPUT_DIRECTORY = "outputs"

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("mode", help="mode of operation", type=str,
                        choices=["train", "predict", "evaluate"])
    PARSER.add_argument("model_name", help="name of the model", type=str, nargs="?",
                        const=None)
    ARGS = PARSER.parse_args()

    if ARGS.mode == "train":
        train(
            batch_size=BATCH_SIZE,
            epoch_count=EPOCH_COUNT,
            warmup_epoch_count=WARMUP_EPOCH_COUNT,
            max_learn_rate=MAX_LEARN_RATE,
            end_learn_rate=END_LEARN_RATE,
            patience=PATIENCE,
            lstm_units=LSTM_UNITS,
            dense_units=DENSE_UNITS,
            dropout=DROPOUT,
            adapter_size=ADAPTER_SIZE,
            optimizer=OPTIMIZER,
            bert_model_path=BERT_MODEL_PATH,
            model_name=ARGS.model_name
        )
    elif ARGS.mode == "predict":
        dev_prediction, test_prediction = predict(
            ARGS.model_name,
            lstm_units=LSTM_UNITS,
            dense_units=DENSE_UNITS,
            dropout=DROPOUT,
            adapter_size=ADAPTER_SIZE,
            optimizer=OPTIMIZER,
            bert_model_path=BERT_MODEL_PATH
        )
        with open(f"{OUTPUT_DIRECTORY}/dev_predictions.tsv", "w") as file_pointer:
            file_pointer.write(
                "\n".join(dev_prediction))
        with open(f"{OUTPUT_DIRECTORY}/test_predictions.tsv", "w") as file_pointer:
            file_pointer.write(
                "\n".join(test_prediction))
    elif ARGS.mode == "evaluate":
        model, data = load_model(
            ARGS.model_name,
            lstm_units=LSTM_UNITS,
            dense_units=DENSE_UNITS,
            dropout=DROPOUT,
            adapter_size=ADAPTER_SIZE,
            optimizer=OPTIMIZER,
            bert_model_path=BERT_MODEL_PATH
        )
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
