import os
import tensorflow as tf
from tensorflow import keras
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

bert_ckpt_dir = os.path.join("model/", "multi_cased_L-12_H-768_A-12")
BERT_CKPT_FILE = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
BERT_CONFIG_FILE = os.path.join(bert_ckpt_dir, "bert_config.json")

def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False


def create_model(
        bert_config_file,
        bert_ckpt_file,
        max_seq_len,
        lstm_units=768,
        dense_units=768,
        dropout=0.5,
        adapter_size=64,
        optimizer=None
):
    optimizer = optimizer or keras.optimizers.Adam()

    # adapter_size = 64  # see - arXiv:1902.00751

    # create the bert layer
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = adapter_size
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32',
                                   name="input_ids")
    output = bert(input_ids)

    print("bert shape", output.shape)
    logits = keras.layers.LSTM(lstm_units, activation="tanh")(output)
    logits = keras.layers.Dropout(dropout)(logits)
    if dense_units:
        logits = keras.layers.Dense(dense_units, activation="tanh")(logits)
        logits = keras.layers.Dropout(dropout)(logits)
    logits = keras.layers.Dense(units=3, activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    # load the pre-trained model weights
    load_stock_weights(bert, bert_ckpt_file)

    # freeze weights if adapter-BERT is used
    if adapter_size is not None:
        freeze_bert_layers(bert)

    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    return model
