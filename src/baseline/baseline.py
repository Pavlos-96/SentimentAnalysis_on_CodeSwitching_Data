import pandas as pd
from pathlib import Path


def distribution(df):
     return df["sentiment"].tolist()

def most_freq(distribution):
    return max(set(distribution), key = distribution.count)

def relative_frequency(distribution, most_frequent):
    freq = 0
    for i in range(len(distribution)):
        if distribution[i] == most_frequent:
            freq += 1
    return freq / len(distribution)

if __name__ == '__main__':
    p = Path.cwd()
    q = p / ".." / ".." / "data"
    data = pd.read_csv(q / "data.csv")

    distr = distribution(data)
    most_frequent = most_freq(distr)
    print("most frequent label: ", most_frequent)
    rel_freq = relative_frequency(distr, most_frequent)
    print("relative frequency: ", rel_freq)