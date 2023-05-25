from data_preparation import load_data_sets
from tqdm import tqdm
import bert
from bert.tokenization.bert_tokenization import FullTokenizer
import numpy as np


class FacebookComments:
    """Processes the data for BERT."""

    DATA_COLUMN = "sentence"
    LABEL_COLUMN = "sentiment"

    def __init__(self, tokenizer: FullTokenizer, sample_size=None, max_seq_len=1024):
        """
        Args:
            tokenizer: the bert-tokenizer
            sample_size
            max_seq_len
        """

        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        train, test, dev = load_data_sets()

        train, test, dev = map(lambda df: df.reindex(
            df[FacebookComments.DATA_COLUMN].str.len().sort_values().index),
                               [train, test, dev])

        if sample_size is not None:
            assert sample_size % 128 == 0
            train, test, dev = train.head(sample_size), test.head(sample_size), \
                               dev.head(sample_size)
            # train, test = map(lambda df: df.sample(sample_size), [train, test])

        ((self.train_x, self.train_y),
         (self.test_x, self.test_y),
         (self.dev_x, self.dev_y)) = map(self._prepare, [train, test, dev])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types),
         (self.dev_x, self.dev_x_token_types)) = map(self._pad,
                                                       [self.train_x, self.test_x, self.dev_x])

    def _prepare(self, df):
        """
        Inputs special tokens, which are needed in BERT

        Args:
            df: dataframe, of the data to be converted

        Returns:
            two arrays for the x and y values with the special tokens
        """

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
        """
        Pads the vectors

        Args:
            ids: arrays of token vectors

        Returns:
            two padded arrays for the x and y values"""

        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)