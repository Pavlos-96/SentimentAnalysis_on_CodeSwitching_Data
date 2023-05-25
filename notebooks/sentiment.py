# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python Hindi English Sentiment
#     language: python
#     name: hi-en-sentiment
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/kpe/bert-for-tf2/blob/master/examples/movie_reviews_with_bert_for_tf2_on_gpu.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] colab_type="text" id="xiYrZKaHwV81"
# This is a modification of https://github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb using the Tensorflow 2.0 Keras implementation of BERT from [kpe/bert-for-tf2](https://github.com/kpe/bert-for-tf2) with the original [google-research/bert](https://github.com/google-research/bert) weights.
#

# + colab={} colab_type="code" id="j0a4mTk9o1Qg"
# Copyright 2019 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# + [markdown] colab_type="text" id="dCpvgG0vwXAZ"
# # Predicting Movie Review Sentiment with [kpe/bert-for-tf2](https://github.com/kpe/bert-for-tf2)
#
# First install some prerequisites:

# + colab={} colab_type="code" id="qFI2_B8ffipb"
# # !pip install tqdm

# + colab={} colab_type="code" id="hsZvic2YxnTz"
import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf


# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="Evlk1N78HIXM" outputId="9ada0ad2-3297-414d-b7bb-748063302382"
# tf.__version__

# + colab={} colab_type="code" id="TAdrQqEccIva"
# if tf.__version__.startswith("1."):
#     tf.enable_eager_execution()


# + [markdown] colab_type="text" id="cp5wfXDx5SPH"
# In addition to the standard libraries we imported above, we'll need to install the [bert-for-tf2](https://github.com/kpe/bert-for-tf2) python package, and do the imports required for loading the pre-trained weights and tokenizing the input text. 

# + colab={} colab_type="code" id="jviywGyWyKsA"
# # !pip install bert-for-tf2

# + colab={} colab_type="code" id="ZtI7cKWDbUVc"
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

# + [markdown] colab_type="text" id="pmFYvkylMwXn"
# # Data

# + [markdown] colab_type="text" id="MC_w8SRqN0fr"
# First, let's download the dataset, hosted by Stanford. The code below, which downloads, extracts, and imports the IMDB Large Movie Review Dataset, is borrowed from [this Tensorflow tutorial](https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub).

# + colab={} colab_type="code" id="fom_ff20gyy6"
from tensorflow import keras
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    f = open(file, "r")

    for line in f:
        line = line.split("\t")
        data["sentence"].append(line[1])
        data["sentiment"].append(line[3].strip())
    df = pd.DataFrame.from_dict(data)
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv("train", index=False)
    test.to_csv("test", index=False)


# import train and test data from csv-files
def load_data_sets(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    return train, test


# + [markdown] colab_type="text" id="CaE2G_2DdzVg"
# Let's use the `MovieReviewData` class below, to prepare/encode 
# the data for feeding into our BERT model, by:
#   - tokenizing the text
#   - trim or pad it to a `max_seq_len` length
#   - append the special tokens `[CLS]` and `[SEP]`
#   - convert the string tokens to numerical `ID`s using the original model's token encoding from `vocab.txt`

# + colab={} colab_type="code" id="2abfwdn-g135"
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer


class FacebookComments:
    DATA_COLUMN = "sentence"
    LABEL_COLUMN = "sentiment"

    def __init__(self, tokenizer: FullTokenizer, sample_size=None, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        train, test = load_data_sets("../data/train_split.csv", "../data/test_split.csv")
        
        train, test = map(lambda df: df.reindex(df[FacebookComments.DATA_COLUMN].str.len().sort_values().index),
                          [train, test])
                
        if sample_size is not None:
            assert sample_size % 128 == 0
            train, test = train.head(sample_size), test.head(sample_size)
            # train, test = map(lambda df: df.sample(sample_size), [train, test])
        
        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad, 
                                                       [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        with tqdm(total=df.shape[0], unit_scale=True) as pbar:
            for ndx, row in df.iterrows():
                text, label = row[FacebookComments.DATA_COLUMN], row[FacebookComments.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len, len(token_ids))
                x.append(token_ids)
                y.append(int(label))
                pbar.update()
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)



# + [markdown] colab_type="text" id="SGL0mEoNFGlP"
# # Uncased BERT
# ## A tweak
#
# Because of a `tf.train.load_checkpoint` limitation requiring list permissions on the google storage bucket, we need to copy the pre-trained BERT weights locally.

# + colab={} colab_type="code" id="lw_F488eixTV"
bert_ckpt_dir="gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12/"
bert_ckpt_file = bert_ckpt_dir + "bert_model.ckpt"
bert_config_file = bert_ckpt_dir + "bert_config.json"

# + colab={"base_uri": "https://localhost:8080/", "height": 566} colab_type="code" id="dGFfkWO07cWG" outputId="773f74c0-0f33-4626-929f-cf35c0996353"
# %%time

bert_model_dir="2018_10_18"
bert_model_name="uncased_L-12_H-768_A-12"

# !mkdir -p .model .model/$bert_model_name

for fname in ["bert_config.json", "vocab.txt", "bert_model.ckpt.meta", "bert_model.ckpt.index", "bert_model.ckpt.data-00000-of-00001"]:
    cmd = f"gsutil cp gs://bert_models/{bert_model_dir}/{bert_model_name}/{fname} .model/{bert_model_name}"
    !$cmd

# !ls -la .model .model/$bert_model_name

# + colab={} colab_type="code" id="049feT8dFprc"
bert_ckpt_dir    = os.path.join(".model/",bert_model_name)
bert_ckpt_file   = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")
# -

# # Multi Cased BERT (ML-BERT)

# Download Pretrained Model
# !wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
# Unzip Model
# !unzip multi_cased_L-12_H-768_A-12.zip
# Move to .model directory
# !mv multi_cased_L-12_H-768_A-12 .model/multi_cased_L-12_H-768_A-12
# Delete .zip file
# !rm multi_cased_L-12_H-768_A-12.zip

# +
# Set ML-BERT as model to use

bert_ckpt_dir    = os.path.join("../model/", "multi_cased_L-12_H-768_A-12")
bert_ckpt_file   = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")
# -

from pathlib import Path
Path.cwd()

# + [markdown] colab_type="text" id="G4xPTleh2X2b"
# # Preparing the Data
#
# Now let's fetch and prepare the data by taking the first `max_seq_len` tokens after tokenizing with the BERT tokenizer, und use `sample_size` examples for both training and testing.

# + [markdown] colab_type="text" id="XA8WHJgzhIZf"
# To keep training fast, we'll take a sample of about 2500 train and test examples, respectively, and use the first 128 tokens only (transformers memory and computation requirements scale quadraticly with the sequence length - so with a TPU you might use `max_seq_len=512`, but on a GPU this would be too slow, and you will have to use a very small `batch_size`s to fit the model into the GPU memory).

# + colab={"base_uri": "https://localhost:8080/", "height": 171} colab_type="code" id="kF_3KhGQ0GTc" outputId="fd993d23-61ce-4aae-c992-5b5591123198"
# %%time

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
data = FacebookComments(tokenizer,
                        sample_size=10*128*2,  #5000,
                        max_seq_len=128)

# + colab={"base_uri": "https://localhost:8080/", "height": 103} colab_type="code" id="prRQM8pDi8xI" outputId="b98433d4-c7e6-4bb5-af54-941f52e0b8e7"
print("            train_x", data.train_x.shape)
print("train_x_token_types", data.train_x_token_types.shape)
print("            train_y", data.train_y.shape)

print("             test_x", data.test_x.shape)

print("        max_seq_len", data.max_seq_len)


# + [markdown] colab_type="text" id="sfRnHSz3iSXz"
# ## Adapter BERT
#
# If we decide to use [adapter-BERT](https://arxiv.org/abs/1902.00751) we need some helpers for freezing the original BERT layers.

# + colab={} colab_type="code" id="IuMOGwFui4it"

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


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler



# + [markdown] colab_type="text" id="ccp5trMwRtmr"
# #Creating a model
#
# Now let's create a classification model using [adapter-BERT](https//arxiv.org/abs/1902.00751), which is clever way of reducing the trainable parameter count, by freezing the original BERT weights, and adapting them with two FFN bottlenecks (i.e. `adapter_size` bellow) in every BERT layer.
#
# **N.B.** The commented out code below show how to feed a `token_type_ids`/`segment_ids` sequence (which is not needed in our case).

# + colab={} colab_type="code" id="6o2a5ZIvRcJq"
from sklearn.metrics import f1_score


def accuracy(original, predicted):
	print("F1 score is: " + str(f1_score(original, predicted, average='macro')))
    
def create_model(max_seq_len, adapter_size=64):
    """Creates a classification model."""

    #adapter_size = 64  # see - arXiv:1902.00751

    # create the bert layer
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = adapter_size
        bert = BertModelLayer.from_params(bert_params, name="bert")
        
    input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    # token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
    # output         = bert([input_ids, token_type_ids])
    output         = bert(input_ids)

    print("bert shape", output.shape)
    #cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    #cls_out = keras.layers.Dropout(0.5)(cls_out)
    # logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.LSTM(768, activation="tanh")(output)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=3, activation="softmax")(logits)

    # model = keras.Model(inputs=[input_ids, token_type_ids], outputs=logits)
    # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    # load the pre-trained model weights
    load_stock_weights(bert, bert_ckpt_file)

    # freeze weights if adapter-BERT is used
    if adapter_size is not None:
        freeze_bert_layers(bert)

    model.compile(optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.summary()
        
    return model


# + colab={"base_uri": "https://localhost:8080/", "height": 517} colab_type="code" id="bZnmtDc7HlEm" outputId="fcd96c78-792c-4032-d188-73a9c21ec304"
adapter_size = None # use None to fine-tune all of BERT
model = create_model(data.max_seq_len, adapter_size=adapter_size)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="ZuLOkwonF-9S" outputId="ce1451d4-310c-41f1-ddd3-a2e0841d8528"
# %%time

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

total_epoch_count = 50
# model.fit(x=(data.train_x, data.train_x_token_types), y=data.train_y,


model.fit(x=data.train_x, y=data.train_y,
          validation_split=0.2,
          batch_size=48,
          shuffle=True,
          epochs=total_epoch_count,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=20,
                                                    total_epoch_count=total_epoch_count),
                     keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                     tensorboard_callback])

model.save_weights('./movie_reviews.h5', overwrite=True)

# %tensorboard --logdir logs/fit

# + colab={"base_uri": "https://localhost:8080/", "height": 120} colab_type="code" id="BSqMu64oHzqy" outputId="95a8284b-b3b2-4a7d-f335-c4ff65f8f920"
# %%time

train_loss, train_accuracy, train_f1_score = model.evaluate(data.train_x, data.train_y)
test_loss, test_accuracy, test_f1_score = model.evaluate(data.test_x, data.test_y)

print("train acc", train_accuracy, "train f1_score", train_f1_score)
print(" test acc", test_accuracy, "test f1_score", test_f1_score)
with open("raw_results.txt", "w") as fp:
    fp.write(f"train acc {train_accuracy} train f1_score {train_f1_score}\n test acc {test_accuracy} test f1_score {test_f1_score}")

# + [markdown] colab_type="text" id="xSKDZEnVabnl"
# # Evaluation
#
# To evaluate the trained model, let's load the saved weights in a new model instance, and evaluate.

# + colab={"base_uri": "https://localhost:8080/", "height": 531} colab_type="code" id="qCpabQ15WS3U" outputId="220b2b55-dd78-4795-d3b6-d364c8323734"
# %%time 

model = create_model(data.max_seq_len, adapter_size=None)
model.load_weights("movie_reviews.h5")

train_loss, train_accuracy, train_f1_score, train_precision, train_recall = model.evaluate(data.train_x, data.train_y)
test_loss, test_accuracy, test_f1_score, test_precision, test_recall = model.evaluate(data.test_x, data.test_y)

print("train acc", train_accuracy, "train f1_score", train_f1_score)
print(" test acc", test_accuracy, "test f1_score", test_f1_score)
with open("results.txt", "w") as fp:
    fp.write(f"train acc {train_accuracy} train f1_score {train_f1_score}\n test acc {test_accuracy} test f1_score {test_f1_score}")

# + [markdown] colab_type="text" id="5uzdOFQ5awM1"
# # Prediction
#
# For prediction, we need to prepare the input text the same way as we did for training - tokenize, adding the special `[CLS]` and `[SEP]` token at begin and end of the token sequence, and pad to match the model input shape.

# + colab={"base_uri": "https://localhost:8080/", "height": 171} colab_type="code" id="m7dAAoCuW1xj" outputId="15f4a565-1659-4c7a-a21a-5a1322389746"
pred_sentences = [
"Bhai movie main social message kya hai batao", # 1
"hamare sallu bhai h", # 1
"kya Sallu dost ko bhul gya kya", # 2
"bhupender panwar hijda musalmaan hai", # 0
"baat online kr rhe the bhai...abhi 5 bje khtm hua h", # 2
]

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
pred_tokens    = map(tokenizer.tokenize, pred_sentences)
pred_tokens    = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

print('pred_token_ids', pred_token_ids.shape)

res = model.predict(pred_token_ids).argmax(axis=-1)

for text, sentiment in zip(pred_sentences, res):
    print(" text:", text)
    print("  res:", sentiment)
