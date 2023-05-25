import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIRECTORY = "data"
RAW_DATA_FILE = Path(DATA_DIRECTORY, "data.txt")
DATA_FILE = Path(DATA_DIRECTORY, "data.csv")
TRAIN_FILE = Path(DATA_DIRECTORY, "train_split.csv")
DEV_FILE = Path(DATA_DIRECTORY, "dev_split.csv")
TEST_FILE = Path(DATA_DIRECTORY, "test_split.csv")


# create dataframe, split data to train and test and save them as csv-files
def split_data():
    """ Creates data split"""

    data = {
        "sentence": [],
        "sentiment": []
    }

    with open(RAW_DATA_FILE) as file_pointer:
        for line in file_pointer:
            line = line.split("\t")
            data["sentence"].append(line[1])
            data["sentiment"].append(line[3].strip())

    df = pd.DataFrame.from_dict(data)
    df.to_csv(DATA_FILE, index=False)
    train, dev_test = train_test_split(df, test_size=0.2, stratify=df["sentiment"])
    train.to_csv(TRAIN_FILE, index=False)
    dev, test = train_test_split(dev_test, test_size=0.5, random_state=42)
    dev.to_csv(DEV_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)


# import train and test data from csv-files
def load_data_sets():
    """Loads data sets"""

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    dev = pd.read_csv(DEV_FILE)
    return train, test, dev


if __name__ == '__main__':
    split_data()
