import atexit
import datetime
import math
import os
from pathlib import Path

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from bert.tokenization.bert_tokenization import FullTokenizer

from data_preparation.facebook_comments import FacebookComments
from model import create_model
from model.evaluate import evaluate

OUTPUT_DIRECTORY = "outputs"
CHECKPOINT_DIRECTORY = os.path.join("model/", "multi_cased_L-12_H-768_A-12")
CHECKPOINT_FILE = "bert_model.ckpt"
CONFIG_FILE = "bert_config.json"
VOCAB_FILE = "vocab.txt"

LOG_DIR = Path("logs/sentiment/", datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))

FILE_WRITER = tf.summary.create_file_writer(f"{LOG_DIR}/metrics")
FILE_WRITER.set_as_default()
atexit.register(FILE_WRITER.close)


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):
    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(
                math.log(end_learn_rate / max_learn_rate) * (
                        epoch - warmup_epoch_count + 1) / (
                        total_epoch_count - warmup_epoch_count + 1))
        res = float(res)
        tf.summary.scalar('learning rate', data=res, step=epoch)
        return res

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,
                                                                       verbose=1)
    return learning_rate_scheduler


def train(
        batch_size=48,
        epoch_count=50,
        warmup_epoch_count=20,
        max_learn_rate=1e-5,
        end_learn_rate=1e-7,
        patience=20,
        lstm_units=768,
        dense_units=768,
        dropout=0.5,
        adapter_size=64,
        optimizer=None,
        bert_model_path=None,
        model_name=None
):
    optimizer = optimizer or keras.optimizers.Adam()
    bert_model_path = bert_model_path or CHECKPOINT_DIRECTORY
    model_name = model_name or datetime.datetime.now().strftime('%Y%m%d-%H%M%s')

    hp.hparams({
        "batch_size": batch_size,
        "epoch_count": epoch_count,
        "warmup_epoch_count": warmup_epoch_count,
        "max_learn_rate": max_learn_rate,
        "end_learn_rate": end_learn_rate,
        "patience": patience,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "dropout": dropout,
        "adapter_size": str(adapter_size),
        "optimizer": optimizer.__class__.__name__
    })

    tokenizer = FullTokenizer(vocab_file=str(Path(bert_model_path, VOCAB_FILE)))
    data = FacebookComments(
        tokenizer,
        sample_size=10 * 128 * 2,
        max_seq_len=128,

    )

    model = create_model(
        os.path.join(bert_model_path, CONFIG_FILE),
        os.path.join(bert_model_path, CHECKPOINT_FILE),
        max_seq_len=data.max_seq_len,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout,
        adapter_size=adapter_size,
        optimizer=optimizer
    )
    # model = create_model(
    #     os.path.join(bert_model_path, CONFIG_FILE),
    #     os.path.join(bert_model_path, CHECKPOINT_FILE),
    #     data.max_seq_len,
    #     adapter_size=None
    # )

    model.fit(
        x=data.train_x,
        y=data.train_y,
        validation_split=0.2,
        batch_size=batch_size,
        shuffle=True,
        epochs=epoch_count,
        callbacks=[
            create_learning_rate_scheduler(
                max_learn_rate=max_learn_rate,
                end_learn_rate=end_learn_rate,
                warmup_epoch_count=warmup_epoch_count,
                total_epoch_count=epoch_count),
            keras.callbacks.EarlyStopping(
                patience=patience,
                restore_best_weights=True),
            keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)])

    test_accuracy, test_f1_score = evaluate(model, data.test_x, data.test_y)

    tf.summary.scalar("test-accuracy", test_accuracy, step=1)
    tf.summary.scalar("test-f1-score", test_f1_score, step=1)

    model.save_weights(
        f"./{OUTPUT_DIRECTORY}/{model_name}.h5",
        overwrite=True)
